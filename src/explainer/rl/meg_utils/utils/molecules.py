import os
import sys
from enum import Enum

import torch
from rdkit import Chem
from rdkit.Chem import RDConfig
from torch_geometric.data import Data

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))


def mol_from_smiles(smiles):
    return Chem.MolFromSmiles(smiles)


def mol_to_smiles(mol):
    return Chem.MolToSmiles(mol)


def atom_valences(atom_types):
    periodic_table = Chem.GetPeriodicTable()
    return [
        max(list(periodic_table.GetValenceList(atom_type))) for atom_type in atom_types
    ]


def check_molecule_validity(mol, transform):
    if type(mol) == Data:
        mol = transform(mol)

    return Chem.SanitizeMol(mol, catchErrors=True) == Chem.SANITIZE_NONE


def mol_to_tox21_pyg(molecule):
    if isinstance(molecule, str):
        molecule = mol_from_smiles(molecule)

    X = torch.nn.functional.one_hot(
        torch.tensor(
            [x_map_tox21[atom.GetSymbol()].value for atom in molecule.GetAtoms()]
        ),
        num_classes=53,
    ).float()

    E = torch.tensor(
        [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in molecule.GetBonds()]
        + [
            [bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]
            for bond in molecule.GetBonds()
        ]
    ).t()

    edge_attr = torch.nn.functional.one_hot(
        torch.tensor(
            [e_map_tox21(bond.GetBondType()) for bond in molecule.GetBonds()]
            + [e_map_tox21(bond.GetBondType()) for bond in molecule.GetBonds()]
        ),
        num_classes=4,
    ).float()

    pyg_mol = Data(x=X, edge_index=E, edge_attr=edge_attr)
    pyg_mol.batch = torch.zeros(X.shape[0]).long()
    pyg_mol.molecule = molecule
    pyg_mol.smiles = mol_to_smiles(molecule)
    return pyg_mol


"""def mol_to_esol_pyg(molecule):
    if isinstance(molecule, str):
        molecule = mol_from_smiles(molecule)

    xs = []
    for atom in molecule.GetAtoms():
        x = []
        x.append(x_map_esol['atomic_num'].index(atom.GetAtomicNum()))
        x.append(x_map_esol['chirality'].index(str(atom.GetChiralTag())))
        x.append(x_map_esol['degree'].index(atom.GetTotalDegree()))
        x.append(x_map_esol['formal_charge'].index(atom.GetFormalCharge()))
        x.append(x_map_esol['num_hs'].index(atom.GetTotalNumHs()))
        x.append(x_map_esol['num_radical_electrons'].index(
            atom.GetNumRadicalElectrons()))
        x.append(x_map_esol['hybridization'].index(
            str(atom.GetHybridization())))
        x.append(x_map_esol['is_aromatic'].index(atom.GetIsAromatic()))
        x.append(x_map_esol['is_in_ring'].index(atom.IsInRing()))
        xs.append(x)

        x = torch.tensor(xs, dtype=torch.float).view(-1, 9)

    edge_indices, edge_attrs = [], []
    for bond in molecule.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        e = []
        e.append(e_map_esol['bond_type'].index(str(bond.GetBondType())))
        e.append(e_map_esol['stereo'].index(str(bond.GetStereo())))
        e.append(e_map_esol['is_conjugated'].index(bond.GetIsConjugated()))

        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]

    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float).view(-1, 3)

    # Sort indices.
    if edge_index.numel() > 0:
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.batch = torch.zeros(x.shape[0]).long()
    data.smiles = mol_to_smiles(molecule)
    return data
"""


def pyg_to_mol_esol(pyg_mol):
    mol = Chem.RWMol()

    X = pyg_mol.x.numpy().tolist()
    X = [Chem.Atom(int(x[0])) for x in X]

    E = pyg_mol.edge_index.t()

    for x in X:
        mol.AddAtom(x)

    for (u, v), attr in zip(E, pyg_mol.edge_attr):
        u = u.item()
        v = v.item()
        attr = attr.numpy().tolist()
        attr = attr[0]

        if mol.GetBondBetweenAtoms(u, v):
            continue

        mol.AddBond(u, v, Chem.BondType.values[attr])

    return mol


def e_map_tox21(bond_type, reverse=False):
    if not reverse:
        if bond_type == Chem.BondType.SINGLE:
            return 0
        elif bond_type == Chem.BondType.DOUBLE:
            return 1
        elif bond_type == Chem.BondType.AROMATIC:
            return 2
        elif bond_type == Chem.BondType.TRIPLE:
            return 3
        else:
            raise Exception("No bond type found")

    if bond_type == 0:
        return Chem.BondType.SINGLE
    elif bond_type == 1:
        return Chem.BondType.DOUBLE
    elif bond_type == 2:
        return Chem.BondType.AROMATIC
    elif bond_type == 3:
        return Chem.BondType.TRIPLE
    else:
        raise Exception("No bond type found")


class x_map_tox21(Enum):
    O = 0  # noqa: E741
    C = 1
    N = 2
    F = 3
    Cl = 4
    S = 5
    Br = 6
    Si = 7
    Na = 8
    I = 9  # noqa: E741
    Hg = 10
    B = 11
    K = 12
    P = 13
    Au = 14
    Cr = 15
    Sn = 16
    Ca = 17
    Cd = 18
    Zn = 19
    V = 20
    As = 21
    Li = 22
    Cu = 23
    Co = 24
    Ag = 25
    Se = 26
    Pt = 27
    Al = 28
    Bi = 29
    Sb = 30
    Ba = 31
    Fe = 32
    H = 33
    Ti = 34
    Tl = 35
    Sr = 36
    In = 37
    Dy = 38
    Ni = 39
    Be = 40
    Mg = 41
    Nd = 42
    Pd = 43
    Mn = 44
    Zr = 45
    Pb = 46
    Yb = 47
    Mo = 48
    Ge = 49
    Ru = 50
    Eu = 51
    Sc = 52
