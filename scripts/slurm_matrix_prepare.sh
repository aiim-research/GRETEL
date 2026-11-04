#!/bin/bash -l
# Freeze the graph pairs of the first passes into a new matrix run, on a CPU node (prepare imports the
# framework and checks every pair, which the login node must not do). Then queue its jobs from the login
# node with `python scripts/paper_probe_matrix.py submit --run-root RUN [--only ...]`, which only calls sbatch.
#
#   sbatch -p <partition> scripts/slurm_matrix_prepare.sh CONFIG RUN_ROOT SOURCE [SOURCE ...]
#
# RUN_ROOT must not exist yet. The log ends with "Matrix: <RUN_ROOT>" when the run is ready.
#SBATCH -J matrix-prepare
#SBATCH -o lab/output/logs/matrix_prepare_%j.out
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=8G
set -euo pipefail
config=${1:?usage: CONFIG RUN_ROOT SOURCE [SOURCE ...]}
run_root=${2:?missing run root}
shift 2
[ $# -gt 0 ] || { echo "missing source"; exit 2; }
for base in "${CONDA_EXE:+$(dirname "$(dirname "$CONDA_EXE")")}" "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3"; do
  if [ -n "$base" ] && [ -f "$base/etc/profile.d/conda.sh" ]; then
    source "$base/etc/profile.d/conda.sh"
    conda activate "${GRETEL_ENV:-GRTL}" && break
  fi
done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
sources=()
for source in "$@"; do sources+=(--source "$source"); done
python scripts/paper_probe_matrix.py prepare --config "$config" --output "$run_root" "${sources[@]}"
