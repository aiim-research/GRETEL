#!/bin/bash -l
# Waits for free memory on the Slurm-assigned GPU (as much as the task's model needs), then runs one probe pass.
#SBATCH -J paper-probes
#SBATCH -p cuda
#SBATCH --gres=gpu:fast:1
#SBATCH --ntasks=1
#SBATCH -c 8
#SBATCH --time=12:00:00
#SBATCH -o lab/output/logs/slurm_%j.out
set -euo pipefail

pass=${1:?usage: generate CONFIG | probe CONFIG RESULTS [PROBE] | matrix-generator/ matrix-judge/ matrix-self RUN_ROOT MODEL_ID | array LIST}
config=${2:?missing config}
# Conda: the first installation found, environment GRETEL_ENV (default GRTL).
for base in "${CONDA_EXE:+$(dirname "$(dirname "$CONDA_EXE")")}" "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3"; do
  if [ -n "$base" ] && [ -f "$base/etc/profile.d/conda.sh" ]; then
    source "$base/etc/profile.d/conda.sh"
    conda activate "${GRETEL_ENV:-GRTL}" && break
  fi
done
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# Thread pools that otherwise size themselves on the whole node: HF tokenizers
# (Rayon) and the torch.compile workers that vLLM starts. srun does not inherit
# -c from sbatch (Slurm >= 22.05), so pass it to the job step explicitly.
export RAYON_NUM_THREADS=$OMP_NUM_THREADS
export TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_COMPILE_THREADS=1
export SRUN_CPUS_PER_TASK=$OMP_NUM_THREADS
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}

chain_role=''
if [ "$pass" = array ]; then
  # One line of a fleet list (scripts/paper_probe_matrix.py fleet or fit) per array task.
  line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$config")
  read -r role second rest <<< "$line"
  if [[ "$second" == /* ]]; then
    # The earlier format: role, run root, model id.
    read -r role run_root model_id <<< "$line"
    pass=matrix-$role
    config=$run_root
    set -- "$pass" "$config" "$model_id"
  else
    # role, GiB of the model (-: none), MiB that must be free, ids, then the runs (fit: configuration, sources).
    read -r role memory need ids rest <<< "$line"
    [ "$memory" = - ] || export GRETEL_GPU_MEMORY_GB=$memory
    MIN_FREE_MB=$need
    read -r -a items <<< "$rest"
    if [ "$role" = fit ]; then
      pass=fit
    else
      pass=chain
      chain_role=$role
      chain_flags=()
      for item in "${items[@]}"; do chain_flags+=(--run-root "$item"); done
    fi
  fi
fi

# 0 while a chain has work that can run now. Reads JSON only.
work_left() {
  local code=0
  python scripts/paper_probe_matrix.py pending --ready --role "$chain_role" --ids "$ids" "${chain_flags[@]}" > /dev/null || code=$?
  [ "$code" != 10 ]
}
if [ -n "$chain_role" ] && ! work_left; then
  echo "$chain_role $ids: nothing to do now (done, or its narrators have not finished)"
  exit 0
fi

if [ "${SKIP_GPU_WAIT:-0}" != 1 ]; then
  gpu=${CUDA_VISIBLE_DEVICES:-0}
  gpu=${gpu%%,*}
  uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader 2> /dev/null | tr -d ' ' || true)
  need=${MIN_FREE_MB:-30000}
  step=${WAIT_STEP:-60}
  # The partition also has 80 GB cards: a model that needs more than the card has would wait for nothing.
  total_mb=$(nvidia-smi -i "$gpu" --query-gpu=memory.total --format=csv,noheader,nounits | tr -d ' ')
  if [ "$need" -gt "$total_mb" ]; then
    echo "$(date '+%F %T') $(hostname) GPU $gpu has $total_mb MiB in all and this model needs $need MiB free:" \
         "queue it on an H200 node (fleet --node <node>)"
    exit 4
  fi
  echo "$(date '+%F %T') $(hostname) GPU $gpu ${uuid}: waiting for $need MiB free"
  # One task of this user at a time checks the memory of a GPU and loads its model on it: two tasks that saw
  # the same free memory would both start and one would fail. The lock is held from the check that passes
  # until the model is loaded (the backend creates GRETEL_GPU_READY), never while waiting.
  locked=0
  if command -v flock > /dev/null; then
    exec 9> "/tmp/gretel-gpu-$(id -un)-${uuid:-$gpu}.lock"
    locked=1
  fi
  waited=0
  shown=-100000
  while true; do
    [ "$locked" = 0 ] || flock 9
    free_mb=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
    change=$((free_mb - shown))
    if [ "${change#-}" -ge 2048 ]; then
      echo "$(date '+%F %T') GPU $gpu: $free_mb MiB free"
      shown=$free_mb
    fi
    [ "$free_mb" -ge "$need" ] && break
    [ "$locked" = 0 ] || flock -u 9
    if [ "$waited" -ge "${MAX_WAIT:-21600}" ]; then
      echo "$(date '+%F %T') GPU $gpu: never $need MiB free in $waited s"
      exit 3
    fi
    sleep "$step"
    waited=$((waited + step))
    # Every half hour: another job may have done the work meanwhile.
    if [ -n "$chain_role" ] && [ $((waited % 1800)) -lt "$step" ] && ! work_left; then
      echo "$chain_role $ids: done by another job while waiting"
      exit 0
    fi
  done
  echo "$(date '+%F %T') GPU $gpu: $free_mb MiB free, starting after $waited s"
  if [ "$locked" = 1 ]; then
    GRETEL_GPU_READY=$(mktemp -u "${TMPDIR:-/tmp}/gretel-ready-XXXXXXXX")
    export GRETEL_GPU_READY
    ( for _ in $(seq 1800); do   # at most an hour
        [ -e "$GRETEL_GPU_READY" ] && break
        kill -0 $$ 2> /dev/null || break
        sleep 2
      done
      flock -u 9
      rm -f "$GRETEL_GPU_READY" ) &
    trap "kill $! 2> /dev/null || true" EXIT
  fi
fi

case "$pass" in
  generate) srun python main.py "$config" "${RUN_ID:-1}" ;;
  probe)
    results=${3:?missing results root}
    extra=()
    [ -z "${4:-}" ] || extra+=(--probe "$4")
    # All dumps unless SAMPLE_LIMIT is set
    [ -z "${SAMPLE_LIMIT:-}" ] || extra+=(--limit "$SAMPLE_LIMIT")
    srun python scripts/run_paper_probes.py --config "$config" --results "$results" "${extra[@]}"
    ;;
  matrix-generator)
    srun python scripts/paper_probe_matrix.py run-generator --run-root "$config" --generator "${3:?missing generator ID}"
    ;;
  matrix-judge)
    srun python scripts/paper_probe_matrix.py run-judge --run-root "$config" --judge "${3:?missing judge ID}"
    ;;
  matrix-self)
    srun python scripts/paper_probe_matrix.py run-self --run-root "$config" --generator "${3:?missing generator ID}"
    ;;
  chain)
    # One model, loaded once, over the runs of the line: a narrator, the judges that share a model, or a self-evaluation.
    srun python scripts/paper_probe_matrix.py run-chain --role "$chain_role" --ids "$ids" "${chain_flags[@]}" 9>&-
    ;;
  fit)
    # Load test of one model of a configuration: items = configuration, then the sources of the pairs.
    fit_flags=()
    for item in "${items[@]:1}"; do fit_flags+=(--source "$item"); done
    srun python scripts/paper_probe_matrix.py run-fit --config "${items[0]}" --model "$ids" "${fit_flags[@]}" 9>&-
    ;;
  *) echo "unknown pass: $pass"; exit 2 ;;
esac
