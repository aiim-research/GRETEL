#!/bin/bash
conda update -n base -c defaults conda -y
conda create -n GRTL python=3.9 -y

conda activate GRTL

#pip install -y torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu


pip install picologging exmol gensim joblib jsonpickle karateclub matplotlib networkx numpy pandas rdkit scikit-learn scipy selfies sqlalchemy black typing-extensions torch_geometric dgl IPython ipykernel flufl.lock jsonc-parser
