#!/usr/bin/env bash
set -euo pipefail

# ==============================
# Configuración
# ==============================

ENV_NAME="GRTL"
PYTHON_VERSION="3.9"

# Token de Hugging Face:
#  - ./setup_grtl_monolitico.sh "hf_XXXX"
#  - HF_TOKEN="hf_XXXX" ./setup_grtl_monolitico.sh
HF_TOKEN="${HF_TOKEN:-${1-}}"

if [[ -z "${HF_TOKEN}" ]]; then
  echo "ERROR: No se ha proporcionado el token de Hugging Face."
  echo "  Usa: HF_TOKEN=\"hf_xxx\" ./setup_grtl_monolitico.sh"
  echo "   o:  ./setup_grtl_monolitico.sh \"hf_xxx\""
  exit 1
fi

# ==============================
# Comprobación de conda
# ==============================

if ! command -v conda &>/dev/null; then
  echo "ERROR: conda no está en el PATH. Instala Miniconda/Anaconda primero."
  exit 1
fi

eval "$(conda shell.bash hook)"

# ==============================
# Actualizar conda base y crear entorno
# ==============================

echo ">>> Actualizando conda base..."
conda update -n base -c defaults conda -y

if conda env list | grep -qE "^[^#]*\b${ENV_NAME}\b"; then
  echo ">>> El entorno ${ENV_NAME} ya existe, se usará tal cual."
else
  echo ">>> Creando entorno ${ENV_NAME} con Python ${PYTHON_VERSION}..."
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi

echo ">>> Activando entorno ${ENV_NAME}..."
conda activate "${ENV_NAME}"

echo ">>> Actualizando pip..."
python -m pip install --upgrade pip

# ==============================
# Núcleo numérico + PyTorch GPU
# ==============================

echo ">>> Instalando NumPy compatible con PyTorch (numpy<2)..."
pip install "numpy<2"

echo ">>> Instalando torch/vision/audio con cu121..."
pip install --force-reinstall \
  torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121

python - << 'EOF'
import torch
print("Torch:", torch.__version__, "CUDA disponible:", torch.cuda.is_available())
EOF

# ==============================
# Paquetes generales del proyecto
# ==============================

echo ">>> Instalando paquetes generales del proyecto..."

pip install \
  picologging==0.9.2 \
  exmol \
  gensim \
  joblib \
  jsonpickle \
  matplotlib \
  networkx \
  pandas \
  scikit-learn \
  scipy \
  selfies \
  sqlalchemy \
  black \
  IPython \
  ipykernel \
  flufl.lock \
  jsonc-parser

# RDKit: vía pip (si falla, luego lo instalas con conda -c rdkit)
pip install rdkit-pypi || echo ">>> AVISO: rdkit vía pip falló, inténtalo luego con 'conda install -c rdkit rdkit'"

# DGL y torch_geometric
pip install dgl || echo ">>> AVISO: dgl vía pip falló, revisa instalación manual."
pip install torch_geometric==2.4.0 || echo ">>> AVISO: torch_geometric==2.4.0 puede no tener wheel perfecta para tu combo torch/cuda."

# ==============================
# NLP stack: spaCy 3.7.x + modelo 3.7.1 (NumPy 1.x)
# ==============================

echo ">>> Instalando spaCy compatible con numpy<2 (spaCy<3.8, thinc<8.3)..."
pip install "thinc<8.3.0" "spacy<3.8.0" nltk

echo ">>> Instalando modelo en_core_web_sm 3.7.1 (compatible spaCy<3.8)..."
pip install \
  "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl"

# ==============================
# Hugging Face: transformers + hub + CLI
# ==============================

echo ">>> Instalando transformers, huggingface_hub (CLI), accelerate, google-genai..."

pip install \
  "huggingface_hub[cli]==0.36.0" \
  "transformers==4.57.1" \
  google-genai \
  accelerate

# ==============================
# Comprobar CLI de Hugging Face
# ==============================

if ! command -v hf &>/dev/null; then
  echo "ERROR: el comando 'hf' no está disponible (huggingface_hub[cli]==0.36.0)."
  exit 1
fi

# ==============================
# Login en Hugging Face (sin prompt)
# ==============================

echo ">>> Autenticando en Hugging Face con hf auth login..."
hf auth login --token "${HF_TOKEN}" --add-to-git-credential

echo ">>> Usuario actual en HF:"
hf auth whoami || true

# ==============================
# Descargar modelo Llama-3.2-1B
# ==============================

echo ">>> Descargando meta-llama/Llama-3.2-1B (original/*) a Llama-3.2-1B..."
hf download meta-llama/Llama-3.2-1B \
  --include "original/*" \
  --local-dir Llama-3.2-1B

echo ">>> TODO LISTO: entorno ${ENV_NAME} GPU configurado."
