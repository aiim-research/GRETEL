from os.path import join

import numpy as np
import pandas as pd
import networkx as nx
from rdkit import Chem
from rdkit.Chem import MolFromSmiles as smi2mol
from rdkit.Chem import MolToSmiles as mol2smi

from src.dataset.generators.base import Generator
from src.dataset.instances.graph import GraphInstance


def smiles2graph(self, smiles, id=-1, label=-1):


    mol, smi, sanitized = sanitize_smiles(smiles)
    if sanitized:
        A,X,W = mol_to_matrices(mol)
        result = GraphInstance(id=id, 
                               label=label, 
                               data=A, 
                               node_features=X, 
                               edge_features=W,
                               graph_features={"smile":smi,"string_repp":smi,"mol":mol})
    else: 
        return None


def sanitize_smiles(self, smiles):
    try:
        mol = smi2mol(smiles, sanitize=True)
        smi_canon = mol2smi(mol, isomericSmiles=False, canonical=True)
        check = smi2mol(smi_canon)
        # if the conversion failed
        if check is None:
            return None, None, False
        return mol, smi_canon, True
    except Exception as e:
        return None, None, False
    
    
def mol_to_matrices(mol):
    n_map = self.dataset.node_features_map
    e_map = self.dataset.edge_features_map
    atms = mol.GetAtoms()
    bnds = mol.GetBonds()
    n = len(atms)
    A = np.zeros((n,n))
    X = np.zeros((n,len(n_map)))
    W = np.zeros((2*len(bnds),len(e_map)))

    for atom in atms:
        i = atom.GetIdx()
        X[i,n_map['Idx']]=i
        X[i,n_map['AtomicNum']]=atom.GetAtomicNum() #TODO: Encode the atomic number as one hot vector (118 Elements in the Table)
        X[i,n_map['FormalCharge']]=atom.GetFormalCharge()
        X[i,n_map['NumExplicitHs']]=atom.GetNumExplicitHs()
        X[i,n_map['IsAromatic']]=int(atom.GetIsAromatic() == True)
        X[i,n_map[atom.GetChiralTag().name]]= 1
        X[i,n_map[atom.GetHybridization().name]]= 1

    p=0
    _p=len(bnds)
    for bond in bnds:
        A[bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()] = 1
        A[bond.GetEndAtomIdx(),bond.GetBeginAtomIdx()] = 1

        W[p,e_map['Conjugated']]=int(bond.GetIsConjugated() == True)
        W[_p,e_map['Conjugated']]=int(bond.GetIsConjugated() == True)

        W[p,e_map[bond.GetBondType().name]] = 1
        W[_p,e_map[bond.GetBondType().name]] = 1

        W[p,e_map[bond.GetBondDir().name]] = 1 
        W[_p,e_map[bond.GetBondDir().name]] = 1

        W[p,e_map[bond.GetStereo().name]] = 1 
        W[_p,e_map[bond.GetStereo().name]] = 1
        p += 1
        _p += 1
    
    return A,X,W