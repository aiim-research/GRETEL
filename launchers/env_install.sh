#!/bin/bash
conda update -n base -c defaults conda -y
conda create -n GRTL python=3.9 -y

conda activate GRTL

#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu


pip install picologging==0.9.2 exmol gensim joblib jsonpickle karateclub matplotlib networkx numpy pandas rdkit scikit-learn scipy selfies sqlalchemy black typing-extensions torch_geometric==2.4.0 dgl IPython ipykernel flufl.lock jsonc-parser
