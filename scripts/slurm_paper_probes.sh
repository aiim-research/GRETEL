#!/bin/bash -l
# Waits for free memory on the Slurm-assigned GPU, then runs one probe pass.
#SBATCH -J paper-probes
#SBATCH -p cuda
#SBATCH --gres=gpu:fast:1
#SBATCH --ntasks=1
#SBATCH -c 8
#SBATCH --time=12:00:00
#SBATCH -o lab/output/logs/slurm_%j.out
set -euo pipefail

pass=${1:?usage: generate CONFIG | probe CONFIG RESULTS [PROBE] | matrix-generator/ matrix-judge RUN_ROOT MODEL_ID}
config=${2:?missing config}
source ~/miniforge3/etc/profile.d/conda.sh
conda activate GRTL
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

gpu=${CUDA_VISIBLE_DEVICES:-0}
gpu=${gpu%%,*}
waited=0
while true; do
  free_mb=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
  echo "GPU $gpu: $free_mb MiB free"
  [ "$free_mb" -ge "${MIN_FREE_MB:-30000}" ] && break
  [ "$waited" -lt "${MAX_WAIT:-21600}" ] || exit 3
  sleep "${WAIT_STEP:-300}"
  waited=$((waited + ${WAIT_STEP:-300}))
done

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
  *) echo "unknown pass: $pass"; exit 2 ;;
esac
