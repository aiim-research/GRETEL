# descarga_llama.py
# descarga_llama_conda.py
from huggingface_hub import snapshot_download
import os, sys
repo_id = "meta-llama/Llama-3.1-8B"
out_dir = "Llama-3.1-8B"

token = "TOKEN_DE_HUG"
if not token:
    raise SystemExit("Exporta HUGGINGFACE_HUB_TOKEN antes: export HUGGINGFACE_HUB_TOKEN=hf_...")

snapshot_download(
    repo_id=repo_id,
    local_dir=out_dir,
    token=token,
)

print("Hecho. Archivos en:", out_dir)
