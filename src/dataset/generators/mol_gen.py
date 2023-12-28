from os.path import join

import numpy as np
import pandas as pd
import networkx as nx
from rdkit import Chem
from rdkit.Chem import MolFromSmiles as smi2mol
from rdkit.Chem import MolToSmiles as mol2smi

from src.dataset.generators.base import Generator
from src.dataset.instances.graph import GraphInstance

class MolGenerator(Generator):

    def init(self):        
        base_path = self.local_config['parameters']['data_dir']
        self._data_file_path = join(base_path, self.local_config['parameters']['data_file_name'])
        self._data_label_name = self.local_config['parameters']['data_label_name']

        self.dataset.node_features_map = rdk_node_features_map()
        self.dataset.edge_features_map = rdk_edge_features_map()
        self.generate_dataset()
        #self.dataset.edge_features_map = {}

    def check_configuration(self):
        super().check_configuration()
        local_config=self.local_config

        if 'data_file_name' not in local_config['parameters']:
            raise Exception("The name of the data file must be given.")
       
        if 'data_label_name' not in local_config['parameters']:
            raise Exception("The name of the label column must be given.")

    def get_num_instances(self):
        return len(self.dataset.instances)
    
    def generate_dataset(self):
        if not len(self.dataset.instances):
            self._read(self._data_file_path)

    def _read(self, path):
        data = pd.read_csv(path)
        data = data.sample(frac=1.0).reset_index(drop=True)
        data_labels = data[self._data_label_name]

        skipped_molecules = 0
        for i in range(len(data)):
            sanitized, g = smile2graph(i - skipped_molecules, data.smiles[i], data_labels[i], self.dataset)
            if sanitized:
                self.dataset.instances.append(g)
            else:
                skipped_molecules += 1
            '''mol, smi, sanitized = self._sanitize_smiles(data.smiles[i])
            if sanitized:
                A,X,W = self.mol_to_matrices(mol)
                self.dataset.instances.append(GraphInstance(id=i - skipped_molecules, 
                                                            label=int(data_labels[i]), 
                                                            data=A, 
                                                            node_features=X, 
                                                            edge_features=W,
                                                            graph_features={"smile":smi,"string_repp":smi,"mol":mol}))
            else:
                skipped_molecules += 1'''

def smile2graph(id, smile, label, dataset):
    mol, smi, sanitized = sanitize_smiles(smile)
    g = None
    if sanitized:
        A,X,W = mol_to_matrices(mol, dataset)
        g = GraphInstance(id=id, 
                        label=int(label), 
                        data=A, 
                        node_features=X, 
                        edge_features=W,
                        graph_features={"smile":smi,"string_repp":smi,"mol":mol})
    return sanitized, g

def mol_to_matrices(mol, dataset):
    n_map = dataset.node_features_map
    e_map = dataset.edge_features_map
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
                
def sanitize_smiles(smiles):
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
    
    

def rdk_node_features_map():
    base_map = {"Idx":0,"AtomicNum":1,"FormalCharge":2,"NumExplicitHs":3,"IsAromatic":4}
    base_map = {**base_map,**rdk_enum_type_to_map(Chem.ChiralType,len(base_map))}
    return {**base_map,**rdk_enum_type_to_map(Chem.HybridizationType,len(base_map))}

def rdk_edge_features_map():
    base_map = {"Conjugated":0}
    base_map = {**base_map,**rdk_enum_type_to_map(Chem.BondType,len(base_map))}
    base_map = {**base_map,**rdk_enum_type_to_map(Chem.BondDir,len(base_map))}
    return {**base_map,**rdk_enum_type_to_map(Chem.BondStereo,len(base_map))}

def rdk_enum_type_to_map(enum_type,offset=0):
    enum_map={}
    for val in enum_type.names:
        enum_map[enum_type.names[val].name]=enum_type.names[val].real+offset
    return enum_map

def rdk_enum_val_to_one_hot(enum_type):
    one_hot_vec = np.zeros((1,len(enum_type.values)))
    one_hot_vec[0,enum_type.real]=1
    return one_hot_vec