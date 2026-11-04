#!/bin/bash -l
# Create the conda environment of the narrative paper's GPU jobs (narrators and judges with vLLM), apart from
# GRTL: the models need a recent vLLM, which brings its own torch and Python >= 3.10, while GRTL (Python 3.9,
# torch 2.5.1) runs the CPU experiments and must keep its versions. The GPU jobs read GRETEL_ENV, so they
# are submitted with GRETEL_ENV=GRTL-llm. Needs network on the node; the versions it got are written to
# lab/output/logs/llm_env_freeze_<job>.txt (pin them with VLLM_VERSION for a later rebuild).
#
#   sbatch -p normal scripts/slurm_setup_llm_env.sh                    # environment GRTL-llm, latest vLLM
#   sbatch -p normal --export=ALL,LLM_ENV=<name>,VLLM_VERSION=<x.y.z> scripts/slurm_setup_llm_env.sh
#SBATCH -J llm-env
#SBATCH -o lab/output/logs/llm_env_%j.out
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
set -euo pipefail
ENV_NAME=${LLM_ENV:-GRTL-llm}
for base in "${CONDA_EXE:+$(dirname "$(dirname "$CONDA_EXE")")}" "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3"; do
  if [ -n "$base" ] && [ -f "$base/etc/profile.d/conda.sh" ]; then source "$base/etc/profile.d/conda.sh"; break; fi
done
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  # conda-forge only: the default Anaconda channels require accepting their terms of service
  conda create -y -n "$ENV_NAME" -c conda-forge --override-channels python=3.12
fi
conda activate "$ENV_NAME"
# pip unpacks torch and the CUDA libraries (several GB) in TMPDIR; /tmp is often small or in RAM
export TMPDIR=${LLM_TMPDIR:-$HOME/.cache/gretel-pip-tmp}
mkdir -p "$TMPDIR"
python -m pip install --upgrade pip
python -m pip install "vllm${VLLM_VERSION:+==$VLLM_VERSION}" torch_geometric rdkit networkx scipy scikit-learn pandas \
    jsonpickle jsonc-parser flufl.lock atpublic psutil pyyaml tqdm python-dateutil sympy
# Every GRETEL module the GPU jobs import: generators and judges, and the dataset and oracle Recourse replays.
CUDA_VISIBLE_DEVICES="" python - <<'PY'
import importlib, sys
sys.path.insert(0, '.')
failed = []
for name in ['vllm', 'torch', 'torch_geometric', 'rdkit', 'scripts.paper_probe_matrix', 'scripts.run_paper_probes',
             'src.utils.context', 'src.LLMexplaneability.huggingface', 'src.utils.probe_generation', 'src.utils.probe_inputs',
             'src.utils.consistency_probe', 'src.utils.recourse_probe', 'src.dataset.dataset_factory',
             'src.oracle.oracle_factory', 'src.dataset.generators.mol_gen', 'src.dataset.generators.TUDataset',
             'src.dataset.generators.treecycles_rand', 'src.oracle.nn.torch', 'src.oracle.custom.oracle_tree_cycles',
             'src.dataset.manipulators.isolated', 'src.dataset.manipulators.oracle_view', 'src.dataset.manipulators.feature_names']:
    try:
        importlib.import_module(name)
    except Exception as exc:
        failed.append(name)
        print('FAILED', name, type(exc).__name__, exc, flush=True)
import torch, vllm
print('python', sys.version.split()[0], 'torch', torch.__version__, 'vllm', vllm.__version__)
print('import check failed:', failed or 'none')
sys.exit(1 if failed else 0)
PY
python -m pip freeze > "lab/output/logs/llm_env_freeze_${SLURM_JOB_ID:-local}.txt"
echo "environment $ENV_NAME ready"
