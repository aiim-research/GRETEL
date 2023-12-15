from src.dataset.instances.base import DataInstance
from src.dataset.dataset_molecular import MolecularDataSet
from src.dataset.data_instance_molecular import MolecularDataInstance

import networkx as nx
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import RDKFingerprint
from rdkit.Chem import rdMolDescriptors
import pandas as pd

class HIVDataset(MolecularDataSet):

    def __init__(self, id, config_dict=None, force_fixed_nodes=False) -> None:
        super().__init__(id, force_fixed_nodes, config_dict)
        self.instances = []
        self.name = 'hiv_dataset'
        self.max_molecule_len = 0
        self.max_n_atoms = 118

    def read_csv_file(self, dataset_path):

        hivdata = pd.read_csv(os.path.join(dataset_path, 'downsampled_HIV.csv'))
        hivdata = hivdata.sample(frac=1.0).reset_index(drop=True)

        skiped_molecules = 0
        for i in range(len(hivdata)):
            mdi = MolecularDataInstance(i - skiped_molecules)
            mdi.name = 'g' + str(i - skiped_molecules)
            mdi.smiles = hivdata.smiles[i]
            mdi.graph_label = int(hivdata.HIV_active[i])
            mdi._max_n_atoms = self.max_n_atoms

            sanitized = mdi.sanitize_smiles(store=True)

            if sanitized:
                self.instances.append(mdi)

                # This adds hydrogen atoms to the molecules that have been kekulized
                m_ext = Chem.AddHs(mdi.molecule)

                if self.max_molecule_len < len(m_ext.GetAtoms()):
                    self.max_molecule_len = len(m_ext.GetAtoms())


        for inst in self.instances:
            inst._max_mol_len = self.max_molecule_len

            



