import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem import MolFromSmiles as smi2mol  # type: ignore
from rdkit.Chem import MolToSmiles as mol2smi  # type: ignore

from src.dataset.instances.base import DataInstance


class MolecularDataInstance(DataInstance):
    def __init__(
        self,
        id=None,
        name: str = None,
        graph=None,
        graph_label: int = None,
        node_labels: dict = None,
        edge_labels: dict = None,
        mcd: int = None,
    ) -> None:
        super().__init__(id, name, graph, graph_label, node_labels, edge_labels, mcd)
        # Create a variable to store the molecular representation of the graph
        self._molecule = None
        self._smiles = None
        self._max_mol_len = 0
        self._max_n_atoms = 0
        self._force_fixed_nodes = False

    @property
    def max_n_nodes(self):
        return self._max_mol_len

    @max_n_nodes.setter
    def max_n_nodes(self, new_val):
        self._max_mol_len = new_val

    @property
    def n_node_types(self):
        return self._max_n_atoms

    @n_node_types.setter
    def n_node_types(self, new_val):
        self._max_n_atoms = new_val

    @property
    def graph(self):
        return self.molecule_to_graph(self._force_fixed_nodes)

    @property
    def molecule(self):
        return self._molecule

    @molecule.setter
    def molecule(self, new_molecule):
        self._molecule = new_molecule
        # The smiles and the molecule should be consistent inside the instance
        self._smiles = Chem.MolToSmiles(new_molecule)

    @property
    def smiles(self):
        return self._smiles

    @smiles.setter
    def smiles(self, new_smiles):
        self._smiles = new_smiles
        # The smiles and the molecule should be consistent inside the instance
        self._molecule = Chem.MolFromSmiles(new_smiles)

    def molecule_to_graph(self, force_fixed_node_positions=False):
        # Creating an empty graph
        G = nx.Graph()

        if not force_fixed_node_positions:
            # For each atom in the molecule create a node in the graph and add the necessary attributes
            for atom in self.molecule.GetAtoms():
                G.add_node(
                    atom.GetIdx(),
                    atomic_num=atom.GetAtomicNum(),
                    is_aromatic=atom.GetIsAromatic(),
                    atom_symbol=atom.GetSymbol(),
                )

            # For each bond between atoms add an edge in the graph with the corresponding attributes
            for bond in self.molecule.GetBonds():
                G.add_edge(
                    bond.GetBeginAtomIdx(),
                    bond.GetEndAtomIdx(),
                    bond_type=bond.GetBondType(),
                )

            return G
        else:
            # For each atom in the molecule create a node in the graph and add the necessary attributes
            for atom in self.molecule.GetAtoms():
                G.add_node(
                    atom.GetIdx(),
                    atomic_num=atom.GetAtomicNum(),
                    is_aromatic=atom.GetIsAromatic(),
                    atom_symbol=atom.GetSymbol(),
                )

            # Add extra dummy atoms present in the molecules dataset
            for i in range(len(self.molecule.GetAtoms()), self.max_n_nodes):
                G.add_node(i, atomic_num=-1, is_aromatic=False, atom_symbol="C")

            # For each bond between atoms add an edge in the graph with the corresponding attributes
            for bond in self.molecule.GetBonds():
                G.add_edge(
                    bond.GetBeginAtomIdx(),
                    bond.GetEndAtomIdx(),
                    bond_type=bond.GetBondType(),
                )

            return G

            # # For each atom in the molecule create a node in the graph and add the necessary attributes
            # atoms = self.molecule.GetAtoms()
            # atoms_in_mol = len(atoms)
            # # iterate from 0 to the max number of atoms in any molecule in the dataset
            # for string_pos in range(0, self._max_mol_len):

            #     # Initialize with an impossible atomic number
            #     atomic_num = 0
            #     # If the current position is lower than the number of atoms in the current molecule
            #     if(string_pos < atoms_in_mol):
            #         atom = atoms[string_pos]
            #         atomic_num = atom.GetAtomicNum()

            #     # Iterate over the possible types of atoms
            #     for atom_i in range(0, self._max_n_atoms):
            #         # If we reach the position matching the atomic number of the atom in the molecule
            #         if atom_i == (atomic_num - 1): # if atomic number is greater than the max_n_atoms this could create a problem
            #             G.add_node( (string_pos * self._max_n_atoms + atom_i),
            #                     atomic_num=atom.GetAtomicNum(),
            #                     is_aromatic=atom.GetIsAromatic(),
            #                     atom_symbol=atom.GetSymbol())

            #         else: # Add a dummy carbon atom
            #             G.add_node( (string_pos * self._max_n_atoms + atom_i),
            #                     atomic_num=6,
            #                     is_aromatic=False,
            #                     atom_symbol='C')

            # # For each bond between atoms add an edge in the graph with the corresponding attributes
            # for bond in self.molecule.GetBonds():
            #     a1 = bond.GetBeginAtomIdx()
            #     a2 = bond.GetEndAtomIdx()
            #     G.add_edge((a1*self._max_n_atoms + atoms[a1].GetAtomicNum()) - 1,
            #                (a2*self._max_n_atoms + atoms[a2].GetAtomicNum()) - 1,
            #                bond_type=bond.GetBondType())

            # return G

    def graph_to_molecule(self, store=True, force_fixed_node_positions=False):
        # If there is no any stored molecule
        if self._molecule is None:
            if not force_fixed_node_positions:
                # Transform the attributted graph into
                molecule = self._att_graph_to_molecule(att_g)

                if store:
                    self._molecule = molecule

                return molecule

            else:
                att_g = self._fixed_nodes_graph_to_att_graph(
                    self.graph, self._max_n_atoms
                )
                molecule = self._att_graph_to_molecule(att_g)

                if store:
                    self._molecule = molecule

                return molecule

        # If there is an stored molecule return that
        return self._molecule

    def _att_graph_to_molecule(self, G):
        # create empty editable mol object
        molecule = Chem.RWMol()

        # Add molecule atoms (nodes)
        for n, n_data in G.nodes(data=True):
            atom = Chem.Atom(n)
            atom.SetAtomicNum(n_data["atomic_num"])
            atom.SetIsAromatic(n_data["is_aromatic"])
            molecule.AddAtom(atom)

        # Add molecule bonds (edges)
        for n1, n2, e_data in G.edges(data=True):
            # If the edge has a bound type use it
            if "bond_type" in e_data:
                bond_type = e_data["bond_type"]
            else:  # In other case just use a Single bound
                bond_type = Chem.rdchem.BondType.SINGLE
            molecule.AddBond(n1, n2, bond_type)

        # Convert RWMol to Mol object
        molecule = molecule.GetMol()

        return molecule

    def _fixed_nodes_graph_to_att_graph(self, fn_G, max_n_atoms):
        result = nx.Graph()

        for n1_ext, n2_ext, e_data in fn_G.edges(data=True):
            # Calculating the id of the node in the molecule
            n1 = int(n1_ext / max_n_atoms)
            if n1 not in result.nodes:
                # Getting node data
                n1_data = self.graph.nodes[n1_ext]

                result.add_node(n1)
                nx.set_node_attributes(result, {n1: n1_data})

            n2 = int(n2_ext / max_n_atoms)
            if n2 not in result.nodes:
                # Getting node data
                n2_data = self.graph.nodes[n2_ext]

                result.add_node(n2)
                nx.set_node_attributes(result, {n2: n2_data})

            result.add_edge(n1, n2)
            result.edges[n1, n2].update(e_data)

        return result

    def sanitize_smiles(self, store=False):
        """Transmfoms the smile of the data instance into a canonical smile representation"""
        mol, smi, result = self._sanitize_smiles()

        if store:
            self._molecule = mol
            self._smiles = smi

        return result

    def _sanitize_smiles(self):
        """Transmfoms the smile of the data instance into a canonical smile representation

        Returns:
        conversion_successful (bool): True/False to indicate if conversion was  successful
        """
        try:
            mol = smi2mol(self._smiles, sanitize=True)
            smi_canon = mol2smi(mol, isomericSmiles=False, canonical=True)
            check = smi2mol(smi_canon)
            # if the conversion failed
            if check is None:
                return None, None, False

            return mol, smi_canon, True

        except Exception:
            return None, None, False

    def to_numpy_arrays(self, store=False, max_n_nodes=-1, n_node_types=-1):
        """Argument for the RD2NX function should be a valid SMILES sequence
        returns: the graph
        """

        # code changed //////////////////////////////////////////////////////
        # m, smi_canon, sanitize_result = self._sanitize_smiles()
        m = self.molecule
        smi_canon = self.smiles
        # ///////////////////////////////////////////////////////////////////

        if self.molecule is None:
            m = Chem.MolFromSmiles(smi_canon)

        # m = Chem.MolFromSmiles(smi_canon)
        m = Chem.AddHs(m)
        order_string = {
            Chem.rdchem.BondType.SINGLE: 1,
            Chem.rdchem.BondType.DOUBLE: 2,
            Chem.rdchem.BondType.TRIPLE: 3,
            Chem.rdchem.BondType.AROMATIC: 4,
        }

        # nodes = np.zeros((N,100))
        # code changed /////////////////////////////////////////////////////
        # nodes = np.zeros((440, 100))
        nodes = np.zeros((self._max_mol_len, self._max_n_atoms))
        # //////////////////////////////////////////////////////////////////

        for i in m.GetAtoms():
            nodes[i.GetIdx(), i.GetAtomicNum()] = 1

        # adj = np.zeros((N,N))
        # code changed //////////////////////////////////////////////////////
        # adj = np.zeros((440, 440))
        adj = np.zeros((self._max_mol_len, self._max_mol_len))
        # ///////////////////////////////////////////////////////////////////
        for j in m.GetBonds():
            u = min(j.GetBeginAtomIdx(), j.GetEndAtomIdx())
            v = max(j.GetBeginAtomIdx(), j.GetEndAtomIdx())
            order = j.GetBondType()
            if order in order_string:
                order = order_string[order]
            else:
                raise Warning("Ignoring bond order" + order)
            adj[u, v] = 1
            adj[v, u] = 1

        # code changed ///////////////////////////////////////////////////////
        # adj += np.eye(440)
        adj += np.eye(self._max_mol_len)
        # ////////////////////////////////////////////////////////////////////

        return nodes, adj
