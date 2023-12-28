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
import exmol

class BBBPDataset(MolecularDataSet):
    """
    This class manages the instances of the human blood-brain barrier (BBB) penetration dataset, 
    provided in
    "A Bayesian Approach to in Silico Blood-Brain Barrier Penetration Modeling
    Ines Filipa Martins, Ana L. Teixeira, Luis Pinheiro, and Andre O. Falcao
    Journal of Chemical Information and Modeling 2012 52 (6), 1686-1697". 
    The molecules are transformed into networkx graphs.
    """

    def __init__(self, id, config_dict=None, force_fixed_nodes=False, exclude_dot_smiles=False) -> None:
        super().__init__(id, force_fixed_nodes, config_dict)
        self.instances = []
        self.name = 'bbbp_dataset'
        self.max_molecule_len = 0
        self.max_n_atoms = 118
        self.exclude_dot_smiles = exclude_dot_smiles

    def __str__(self):
        return self.__class__.__name__
    
    def read_molecules_file(self, dataset_path):
        """
        Reads the dataset from the txt file
        """

        dataset_uri = os.path.join(dataset_path, 'bbbp_no_dots.txt')
        instance_id = 0
        max_mol_len=0
        instances = []

        # Open the dataset
        with open(dataset_uri, 'r') as data_reader:
            # Read the molecular instance in each line
            for line in data_reader.readlines():
                fields = line.split()

                # Create a record to store the instance data
                i_data = {'id': None, 'name': None, 'label': None, 'smiles': None}
                
                # Avoid reading the first line
                if fields[0] != 'num':
                    # Getting the instance number from the dataset (We are going to use our own instance number instead)
                    instance_number = fields[0]

                    # Reading the instance name
                    instance_name = fields[1]
                    label_position = 2
                    while (fields[label_position] != 'p' and fields[label_position] != 'np'):
                        instance_name = instance_name + fields[label_position]
                        label_position+=1
                    instance_name = instance_name.replace(',', '-')
                    instance_name = instance_name.replace('"', '')
                    instance_name = instance_name.replace('/', '')
                    instance_name = instance_name.replace('\\', '')
                    instance_name = instance_name.replace('?', '')

                    # Reading the instance label
                    # class 1 if penetrates the brain (p) or 0 if does not penetrate the brain (np)
                    instance_label = 1 if fields[label_position] == 'p' else 0

                    # Reading the instance smiles
                    instance_smiles = fields[label_position + 1]
                    instance_molecule = Chem.MolFromSmiles(instance_smiles)
                    skip_molecule = False

                    if instance_molecule is None:
                        skip_molecule = True

                    if '.' in instance_smiles and self.exclude_dot_smiles:
                        skip_molecule = True

                    # Checking that it was possible to generate a molecule from the smiles
                    if not skip_molecule:
                        i_data['id'] = instance_id
                        i_data['name'] = 'g' + str(instance_id) + '_' + instance_name
                        i_data['label'] = instance_label
                        i_data['smiles'] = instance_smiles
                        instance_id +=1

                        instances.append(i_data)

                        # Getting the atoms in the molecule and the max length among all the molecules
                        molecule_atoms = instance_molecule.GetAtoms()

                        atoms_in_instance = len(molecule_atoms)
                        if atoms_in_instance > max_mol_len:
                            max_mol_len = atoms_in_instance

        # storing the max molecule lenght
        self.max_molecule_len = max_mol_len
        # Creating the instances and storing them
        skipped_instances = 0                
        for i in instances:
            data_instance = MolecularDataInstance(i['id'] - skipped_instances)
            data_instance._max_mol_len = self.max_molecule_len
            data_instance._max_n_atoms = self.max_n_atoms
            data_instance._force_fixed_nodes = self.force_fixed_nodes
            data_instance.name = i['name']
            data_instance.graph_label = i['label']
            data_instance.smiles = i['smiles']

            san_success = data_instance.sanitize_smiles()

            if san_success:
                # Store the data instance
                self.instances.append(data_instance)
            else:
                skipped_instances += 1


    def read_csv_file(self, dataset_path):

        bbbp_data = pd.read_csv(os.path.join(dataset_path, 'BBBP.csv'))
        bbbp_data = bbbp_data.sample(frac=1.0).reset_index(drop=True)

        skiped_molecules = 0
        for i in range(len(bbbp_data)):
            mdi = MolecularDataInstance(i - skiped_molecules)
            mdi.name = 'g' + str(i - skiped_molecules)
            mdi.smiles = bbbp_data.smiles[i]
            mdi.graph_label = int(bbbp_data.p_np[i])
            mdi._max_n_atoms = self.max_n_atoms
            mdi._force_fixed_nodes = self.force_fixed_nodes

            sanitized = mdi.sanitize_smiles(store=True)

            if sanitized:
                self.instances.append(mdi)

                # This adds hydrogen atoms to the molecules that have been kekulized
                m_ext = Chem.AddHs(mdi.molecule)

                if self.max_molecule_len < len(list(m_ext.GetAtoms())):
                    self.max_molecule_len = len(list(m_ext.GetAtoms()))
            else:
                skiped_molecules += 1


        for inst in self.instances:
            inst._max_mol_len = self.max_molecule_len            