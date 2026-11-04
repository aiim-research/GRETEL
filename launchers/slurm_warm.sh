#!/bin/bash -l
# Build the cached objects of configs (dataset and folds, oracle, DCM, ...) on a compute node without
# running the experiment (tools/warm_caches.py), for the caches too slow to build on a laptop. Bring them
# back and commit them before any queue that needs them (docs/rerun/README.md).
#
#   sbatch -p <partition> launchers/slurm_warm.sh <config> [<config> ...]
#
#SBATCH -J warm
#SBATCH -o output/slurm/%x_%j.out
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=8G

set -uo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} \
       MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
ENV_NAME=${GRETEL_ENV:-GRTL}
for base in "${CONDA_EXE:+$(dirname "$(dirname "$CONDA_EXE")")}" "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3"; do
    if [ -n "$base" ] && [ -f "$base/etc/profile.d/conda.sh" ]; then
        source "$base/etc/profile.d/conda.sh"
        conda activate "$ENV_NAME" && break
    fi
done
echo "# job ${SLURM_JOB_ID:-?} on $(hostname), started $(date '+%Y-%m-%d %H:%M:%S')"
CUDA_VISIBLE_DEVICES="" python tools/warm_caches.py "$@"
STATUS=$?
echo "# finished $(date '+%Y-%m-%d %H:%M:%S') with exit status $STATUS"
exit $STATUS
