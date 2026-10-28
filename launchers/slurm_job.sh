#!/bin/bash -l
# One GRETEL config per Slurm array task. Submitted by launchers/slurm_submit.sh, which passes the
# list of pending configs; task i runs line i+1 of that list. Never edit configs on the server: change
# them locally, commit, push, and `git pull` there.
#
#SBATCH -J gretel
#SBATCH -o output/logs/%x_%A_%a.out
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=8G

set -uo pipefail

LIST="$1"
CFG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$LIST")
if [ -z "$CFG" ]; then
    echo "no config at line $((SLURM_ARRAY_TASK_ID + 1)) of $LIST"; exit 1
fi

# One thread pool per CPU granted to the task (torch, numpy and BLAS otherwise grab every core of the
# node and the jobs on it slow each other down).
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

# Conda: activate GRETEL_ENV if given (default GRTL); otherwise keep the environment exported from the
# shell that ran sbatch.
ENV_NAME=${GRETEL_ENV:-GRTL}
for base in "${CONDA_EXE:+$(dirname "$(dirname "$CONDA_EXE")")}" "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3"; do
    if [ -n "$base" ] && [ -f "$base/etc/profile.d/conda.sh" ]; then
        source "$base/etc/profile.d/conda.sh"
        conda activate "$ENV_NAME" && break
    fi
done

echo "# job ${SLURM_ARRAY_JOB_ID:-?}_${SLURM_ARRAY_TASK_ID:-?} on $(hostname), partition ${SLURM_JOB_PARTITION:-?}, ${SLURM_CPUS_PER_TASK:-?} cpus"
echo "# python $(which python)  conda env ${CONDA_DEFAULT_ENV:-none}"
echo "# config $CFG"
echo "# started $(date '+%Y-%m-%d %H:%M:%S')"

python scripts/run_experiments.py --one "$CFG" --run-number 1
STATUS=$?
echo "# finished $(date '+%Y-%m-%d %H:%M:%S') with exit status $STATUS"
exit $STATUS
