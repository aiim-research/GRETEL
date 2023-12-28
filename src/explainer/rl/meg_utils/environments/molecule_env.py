"""
Defines the Markov decision process of generating a molecule.
The problem of molecule generation as a Markov decision process, the
state space, action space, and reward function are defined.

"""
import copy
import itertools
import numpy as np

from rdkit import Chem
from rdkit.Chem import Draw
from src.explainer.rl.meg_utils.utils.molecules import mol_from_smiles
from src.explainer.rl.meg_utils.environments.base_env import Result, BaseEnvironment          
          
class Molecule(BaseEnvironment):
    
    """ 
        Defines the Markov decision process of generating a molecule.
    """
    
    def __init__(self,
                 allow_removal=True,
                 allow_node_addition=True,
                 allow_edge_addition=True,
                 allow_no_modification=True,
                 allow_bonds_between_rings=True,
                 allowed_ring_sizes=None,
                 target_fn=None,
                 max_steps=10,
                 record_path=False):
        
        super(Molecule, self).__init__(target_fn=target_fn,
                                       max_steps=max_steps)
        
        self.allow_removal = allow_removal
        self.allow_node_addition = allow_node_addition
        self.allow_edge_addition = allow_edge_addition
        self.allow_no_modification = allow_no_modification
        self.allow_bonds_between_rings = allow_bonds_between_rings
        self.allowed_ring_sizes = allowed_ring_sizes
        self._valid_actions = []
        # The status should be 'terminated' if initialize() is not called.
        self.record_path = record_path
        self._path = []
        self._max_bonds = 4
        self.action_counter = 1

        
    def get_path(self):
        return self._path
    
    def set_instance(self, new_instance):
        self._init_instance = new_instance
        self.atom_types = np.unique(
            [x.GetSymbol() for x in mol_from_smiles(self._init_instance.smiles).GetAtoms()]
        )
        atom_types = list(self.atom_types)
        self._max_new_bonds = dict(
            list(zip(atom_types, self.atom_valences(atom_types))))

    
    def initialize(self):
        """Resets the MDP to its initial state."""
        self._state = self.init_instance
        if self.record_path:
            self._path = [self._state]
        self._valid_actions = self.get_valid_actions(force_rebuild=True)
        self._counter = 0
        self.action_counter = 1

    def get_valid_actions(self, state=None, force_rebuild=False):
        if state is None:
            if self._valid_actions and not force_rebuild:
                return copy.deepcopy(self._valid_actions)
            state = self._state
        if isinstance(state, Chem.Mol):
            state = Chem.MolToSmiles(state)
        self._valid_actions = self.__get_valid_actions(
            state,
            atom_types=self.atom_types,
            allow_removal=self.allow_removal,
            allow_no_modification=self.allow_no_modification,
            allowed_ring_sizes=self.allowed_ring_sizes,
            allow_bonds_between_rings=self.allow_bonds_between_rings)
        return copy.deepcopy(self._valid_actions)
    
    def __get_valid_actions(self, state, atom_types,
                            allow_removal=True, allow_no_modification=True,
                            allowed_ring_sizes=None, allow_bonds_between_rings=True):
        if not state:
            print("I might be here")
            return copy.deepcopy(atom_types)
    
        mol = Chem.MolFromSmiles(state.smiles)
        if not mol:
            raise ValueError(f'Recieved invalid state {state}')
        
        atom_valences = {
            atom_type: self.atom_valences([atom_type])[0] for atom_type in atom_types
            }
        atoms_with_free_valence = {}
        for i in range(1, max(atom_valences.values())):
            # Only atoms that allow us to replace at least one H with a new bond are
            # # enumerated here
            atoms_with_free_valence[i] = [
                atom.GetIdx() for atom in mol.GetAtoms() if atom.GetNumImplicitHs() >= i
            ]
        
        valid_actions = set()
        valid_actions.update(
            self._atom_additions(
                mol,
                atom_types=atom_types,
                atom_valences=atom_valences,
                atoms_with_free_valence=atoms_with_free_valence
            )
        )
        valid_actions.update(
            self._bond_addition(
                mol,
                atoms_with_free_valence=atoms_with_free_valence,
                allowed_ring_sizes=allowed_ring_sizes,
                allow_bonds_between_rings=allow_bonds_between_rings
            )
        )
        
        if allow_removal:
            valid_actions.update(self._bond_removal(mol))
        # add the same state
        if allow_no_modification:
            valid_actions.add(state)
        return valid_actions
    
    
    def _atom_additions(self, state, atom_types,
                        atom_valences, atoms_with_free_valence):
        bond_order = {
            1: Chem.BondType.SINGLE,
            2: Chem.BondType.DOUBLE,
            3: Chem.BondType.TRIPLE
        }
        
        atom_addition = set()
        for i in bond_order:
            for atom in atoms_with_free_valence[i]:
                for element in atom_types:
                    if atom_valences[element] >= i:
                        new_state = Chem.RWMol(state)
                        idx = new_state.AddAtom(Chem.Atom(element))
                        new_state.AddBond(atom, idx, bond_order[i])
                        sanitization_result = Chem.SanitizeMol(new_state, catchErrors=True)
                        # when sanitization fails
                        if sanitization_result:
                            continue
                        data_instance = MolecularDataInstance(self._state.id + self.action_counter)
                        data_instance.molecule = new_state
                        data_instance._max_mol_len = self.init_instance._max_mol_len
                        data_instance._max_n_atoms = self.init_instance._max_n_atoms
                        atom_addition.add(data_instance)
                        self.action_counter += 1
        return atom_addition


    def _bond_addition(self, state,
                       atoms_with_free_valence,
                       allowed_ring_sizes,
                       allow_bonds_between_rings):
        bond_orders = [
            None,
            Chem.BondType.SINGLE,
            Chem.BondType.DOUBLE,
            Chem.BondType.TRIPLE,
        ]
        
        bond_addition = set()
        for valence, atoms in atoms_with_free_valence.items():
            for atom1, atom2 in itertools.combinations(atoms, 2):
                # Get the bond from a copy of the molecule so that SetBondType() doesn't
                # modify the original state.
                bond = Chem.Mol(state).GetBondBetweenAtoms(atom1, atom2)
                new_state = Chem.RWMol(state)
                # Kekulize the new state to avoid sanitization errors; note that bonds
                # that are aromatic in the original state are not modified (this is
                # enforced by getting the bond from the original state with
                # GetBondBetweenAtoms()).
                Chem.Kekulize(new_state, clearAromaticFlags=True)
                if bond:
                    if bond.GetBondType() not in bond_orders:
                        continue  # Skip aromatic bonds.
                    idx = bond.GetIdx()
                    # Compute the new bond order as an offset from the current bond order.
                    bond_order = bond_orders.index(bond.GetBondType())
                    bond_order += valence
                    if bond_order < len(bond_orders):
                        idx = bond.GetIdx()
                        bond.SetBondType(bond_orders[bond_order])
                        new_state.ReplaceBond(idx, bond)
                    else:
                        continue
                # If do not allow new bonds between atoms already in rings.
                elif (not allow_bonds_between_rings and
                        (state.GetAtomWithIdx(atom1).IsInRing() and
                        state.GetAtomWithIdx(atom2).IsInRing())):
                    continue
                # If the distance between the current two atoms is not in the
                # allowed ring sizes
                elif (allowed_ring_sizes is not None and
                        len(Chem.rdmolops.GetShortestPath(
                            state, atom1, atom2)) not in allowed_ring_sizes):
                    continue
                else:
                    new_state.AddBond(atom1, atom2, bond_orders[valence])
                sanitization_result = Chem.SanitizeMol(new_state, catchErrors=True)
                # When sanitization fails
                if sanitization_result:
                    continue
                
                data_instance = MolecularDataInstance(self._state.id + self.action_counter)
                data_instance.molecule = new_state
                data_instance._max_mol_len = self.init_instance._max_mol_len
                data_instance._max_n_atoms = self.init_instance._max_n_atoms
                bond_addition.add(data_instance)
                self.action_counter += 1
                
        return bond_addition

    def _bond_removal(self, state):
        bond_orders = [
            None,
            Chem.BondType.SINGLE,
            Chem.BondType.DOUBLE,
            Chem.BondType.TRIPLE,
        ]
        bond_removal = set()
        for valence in [1, 2, 3]:
            for bond in state.GetBonds():
                # Get the bond from a copy of the molecule so that SetBondType() doesn't
                # modify the original state.
                bond = Chem.Mol(state).GetBondBetweenAtoms(bond.GetBeginAtomIdx(),
                                                            bond.GetEndAtomIdx())
                if bond.GetBondType() not in bond_orders:
                    continue  # Skip aromatic bonds.
                new_state = Chem.RWMol(state)
                # Kekulize the new state to avoid sanitization errors; note that bonds
                # that are aromatic in the original state are not modified (this is
                # enforced by getting the bond from the original state with
                # GetBondBetweenAtoms()).
                Chem.Kekulize(new_state, clearAromaticFlags=True)
                # Compute the new bond order as an offset from the current bond order.
                bond_order = bond_orders.index(bond.GetBondType())
                bond_order -= valence
                if bond_order > 0:  # Downgrade this bond.
                    idx = bond.GetIdx()
                    bond.SetBondType(bond_orders[bond_order])
                    new_state.ReplaceBond(idx, bond)
                    sanitization_result = Chem.SanitizeMol(new_state, catchErrors=True)
                    # When sanitization fails
                    if sanitization_result:
                        continue
                    data_instance = MolecularDataInstance(self._state.id + self.action_counter)
                    data_instance.molecule = new_state
                    data_instance._max_mol_len = self.init_instance._max_mol_len
                    data_instance._max_n_atoms = self.init_instance._max_n_atoms
                    bond_removal.add(data_instance)
                    self.action_counter += 1
                elif bond_order == 0:  # Remove this bond entirely.
                    atom1 = bond.GetBeginAtom().GetIdx()
                    atom2 = bond.GetEndAtom().GetIdx()
                    new_state.RemoveBond(atom1, atom2)
                    sanitization_result = Chem.SanitizeMol(new_state, catchErrors=True)
                    # When sanitization fails
                    if sanitization_result:
                        continue
                    smiles = Chem.MolToSmiles(new_state)
                    parts = sorted(smiles.split('.'), key=len)
                    # We define the valid bond removing action set as the actions
                    # that remove an existing bond, generating only one independent
                    # molecule, or a molecule and an atom.
                    if len(parts) == 1 or len(parts[0]) == 1:
                        data_instance = MolecularDataInstance(self._state.id + self.action_counter)
                        data_instance.smiles = parts[-1]
                        data_instance._max_mol_len = self.init_instance._max_mol_len
                        data_instance._max_n_atoms = self.init_instance._max_n_atoms
                        bond_removal.add(data_instance)
                        self.action_counter += 1
        return bond_removal
    
    def reward(self):
        return 0.0
    
    def step(self, action):
        if self._counter >= self.max_steps or self.goal_reached():
            raise ValueError('This episode is terminated.')
        if action.id not in [inst.id for inst in self._valid_actions]:
            print(action)
            raise ValueError('Invalid action.')
        self._state = action
        if self.record_path:
            self._path.append(self._state)
        self._valid_actions = self.get_valid_actions(force_rebuild=True)
        self._counter += 1

        result = Result(
            state=self._state,
            reward=self.reward(),
            terminated=(self._counter >= self.max_steps) or self.goal_reached())
        
        return result
    
    def visualize_state(self, state=None, **kwargs):
        if state is None:
            state = self._state
        if isinstance(state, str):
            state = Chem.MolFromSmiles(state)
        return Draw.MolToImage(state, **kwargs)
    
    def atom_valences(self, atom_types):
        periodic_table = Chem.GetPeriodicTable()
        return [
            max(list(periodic_table.GetValenceList(atom_type)))
            for atom_type in atom_types
        ]