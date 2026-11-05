#!/bin/bash -l
# Download the exact checkpoints of a matrix configuration into HF_HOME (default ~/hf_cache, the one
# scripts/slurm_paper_probes.sh reads with HF_HUB_OFFLINE=1). Needs network on the compute node and,
# for gated models (Llama), the license accepted on huggingface.co and a token (`HF_TOKEN` exported,
# or `hf auth login` done once in GRTL-llm). Already complete snapshots are skipped.
#
#   sbatch -p <partition> scripts/slurm_hf_download.sh lab/config/narratives/matrix.json
#SBATCH -J hf-download
#SBATCH -o lab/output/logs/hf_download_%j.out
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
set -euo pipefail
config=${1:?usage: MATRIX_CONFIG}
for base in "${CONDA_EXE:+$(dirname "$(dirname "$CONDA_EXE")")}" "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3"; do
  if [ -n "$base" ] && [ -f "$base/etc/profile.d/conda.sh" ]; then
    source "$base/etc/profile.d/conda.sh"
    conda activate "${GRETEL_ENV:-GRTL-llm}" && break
  fi
done
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
# the token written by `hf auth login` lives in the default cache, not in HF_HOME
[ -f "$HOME/.cache/huggingface/token" ] && export HF_TOKEN_PATH=${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}
python - "$config" <<'PY'
import json, sys
from huggingface_hub import snapshot_download
config = json.load(open(sys.argv[1]))
failed = []
for entry in config['generators'] + config['judges']:
    if entry.get('class') or entry.get('cpu'):
        continue   # not a Hugging Face model (the synthetic narrators of V0)
    model, revision = entry['parameters']['model'], entry['parameters'].get('revision')
    try:
        path = snapshot_download(model, revision=revision, allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt', '*.jinja', '*.py'],
                                     ignore_patterns=['original/*', 'metal/*'])
        print('ok', model, revision, path, flush=True)
    except Exception as exc:
        failed.append(model)
        print('FAILED', model, revision, type(exc).__name__, exc, flush=True)
print('failed:', failed or 'none')
sys.exit(1 if failed else 0)
PY
