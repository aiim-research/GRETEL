#!/bin/bash -l
#SBATCH -J pretrain
#SBATCH -o output/slurm/%x_%j.out
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=8G
# Build the selector checkpoints of the given datasets from their collections on a compute node
# (scripts/pretrain_selector_checkpoints.py; it skips checkpoints that exist and collections that are
# missing). Used by launchers/slurm_selector_arm.sh between the collections and the selector arms.
# The checkpoints come back with `make server-import` and are committed from the local clone.
#
#   sbatch -p <partition> [--dependency=afterany:<job>] launchers/slurm_pretrain.sh <dataset> [<dataset> ...]
#
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
echo "# job ${SLURM_JOB_ID:-?} on $(hostname), started $(date '+%Y-%m-%d %H:%M:%S'): $*"
CUDA_VISIBLE_DEVICES="" python scripts/pretrain_selector_checkpoints.py --datasets "$@"
STATUS=$?
echo "# finished $(date '+%Y-%m-%d %H:%M:%S') with exit status $STATUS"
exit $STATUS
