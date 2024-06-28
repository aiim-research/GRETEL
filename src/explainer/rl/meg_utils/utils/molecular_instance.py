import numpy as np

from rdkit import Chem
from rdkit.Chem import MolFromSmiles as smi2mol  # type: ignore
from rdkit.Chem import MolToSmiles as mol2smi  # type: ignore

from src.dataset.instances.graph import GraphInstance


class MolecularInstance(GraphInstance):
    def __init__(
        self,
        id,
        label,
        data,
        node_features=None,
        edge_features=None,
        edge_weights=None,
        graph_features=None,
        dataset=None,
    ):
        super().__init__(
            id,
            label,
            data,
            node_features,
            edge_features,
            edge_weights,
            graph_features,
            dataset,
        )

    @staticmethod
    def from_graph_instance(graph_instance: GraphInstance):
        return MolecularInstance(
            graph_instance.id,
            graph_instance.label,
            graph_instance.data,
            graph_instance.node_features,
            graph_instance.edge_features,
            graph_instance.edge_weights,
            graph_instance.graph_features,
            graph_instance._dataset,
        )

    @property
    def molecule(self):
        return self.graph_features.get("mol")

    @molecule.setter
    def molecule(self, new_molecule):
        self._update_molecule(new_molecule)
        try:
            smiles = mol2smi(self.molecule, isomericSmiles=False, canonical=True)
            self._update_smiles(smiles)
        except RuntimeError as e:
            pass
        self._update_graph_from_molecule()

    @property
    def smiles(self):
        return self.graph_features["smile"]

    @smiles.setter
    def smiles(self, new_smiles):
        self._update_smiles(new_smiles)
        molecule = smi2mol(self.smiles, sanitize=True)
        self._update_molecule(molecule)
        self._update_graph_from_molecule()

    def _update_molecule(self, new_molecule):
        self.graph_features["mol"] = new_molecule

    def _update_smiles(self, new_smiles):
        self.graph_features["smile"] = new_smiles
        self.graph_features["string_repp"] = new_smiles

    def _update_graph_from_molecule(self):
        n_map = self._dataset.node_features_map
        e_map = self._dataset.edge_features_map
        atms = self.molecule.GetAtoms()
        bnds = self.molecule.GetBonds()
        n = len(atms)
        A = np.zeros((n, n))
        X = np.zeros((n, len(n_map)))
        W = np.zeros((2 * len(bnds), len(e_map)))

        for atom in atms:
            i = atom.GetIdx()
            X[i, n_map["Idx"]] = i
            X[i, n_map["AtomicNum"]] = (
                atom.GetAtomicNum()
            )  # TODO: Encode the atomic number as one hot vector (118 Elements in the Table)
            X[i, n_map["FormalCharge"]] = atom.GetFormalCharge()
            X[i, n_map["NumExplicitHs"]] = atom.GetNumExplicitHs()
            X[i, n_map["IsAromatic"]] = int(bool(atom.GetIsAromatic()))
            X[i, n_map[atom.GetChiralTag().name]] = 1
            X[i, n_map[atom.GetHybridization().name]] = 1

        p = 0
        _p = len(bnds)
        for bond in bnds:
            A[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] = 1
            A[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()] = 1

            W[p, e_map["Conjugated"]] = int(bool(bond.GetIsConjugated()))
            W[_p, e_map["Conjugated"]] = int(bool(bond.GetIsConjugated()))

            W[p, e_map[bond.GetBondType().name]] = 1
            W[_p, e_map[bond.GetBondType().name]] = 1

            W[p, e_map[bond.GetBondDir().name]] = 1
            W[_p, e_map[bond.GetBondDir().name]] = 1

            W[p, e_map[bond.GetStereo().name]] = 1
            W[_p, e_map[bond.GetStereo().name]] = 1
            p += 1
            _p += 1

        self.data = A
        self.node_features = X
        self.edge_features = W
        self.edge_weights = np.ones(len(np.nonzero(A)[0]))
        # self.edge_weights = self.__init_edge_weights(None).astype(np.float32)

    def __eq__(self, other):
        if isinstance(other, MolecularInstance):
            return self.molecule == other.molecule
        return False

    def __hash__(self):
        return hash(self.molecule)

    def __deepcopy__(self, memo):
        graph = GraphInstance.__deepcopy__(self, memo)
        graph = MolecularInstance.from_graph_instance(graph)
        graph.molecule = Chem.RWMol(self.molecule)
        return graph
