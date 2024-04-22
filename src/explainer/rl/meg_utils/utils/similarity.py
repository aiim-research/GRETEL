from rdkit import DataStructs
from torch.nn import functional as F
from rdkit.Chem import AllChem
from src.explainer.rl.meg_utils.utils.fingerprints import Fingerprint

def tanimoto_similarity(fp1, fp2):
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def cosine_similarity(encoding_a, encoding_b):
    return F.cosine_similarity(encoding_a, encoding_b).item()

def rescaled_cosine_similarity(molecule_a, molecule_b, S, scale="mean"):
    value = cosine_similarity(molecule_a, molecule_b)

    max_ = 1
    min_ = min(S) if scale == "min" else sum(S) / len(S)

    return (value - min_) / (max_ - min_)

def get_similarity(name, model, fp_len=None, fp_rad=None):
    if name == "tanimoto":
        similarity = lambda x, y: tanimoto_similarity(x, y)
        make_encoding = lambda x: Fingerprint(AllChem.GetMorganFingerprintAsBitVect(x.molecule, fp_len, fp_rad), fp_len)

    """elif name == "rescaled_neural_encoding":
        similarity = lambda x, y: rescaled_cosine_similarity(x, y, similarity_set)

        make_encoding = lambda x: model(x.x, x.edge_index)[1]
        original_encoding = make_encoding(original_molecule)

    elif name == "neural_encoding":
        similarity = lambda x, y: cosine_similarity(x, y)

        make_encoding = lambda x: model(x.x, x.edge_index)[1][1]
        original_encoding = make_encoding(original_molecule)

    elif name == "combined":
        similarity = lambda x, y: 0.5 * cosine_similarity(x[0], y[0]) + 0.5 * tanimoto_similarity(x[1], y[1])

        make_encoding = lambda x: (model(x.x, x.edge_index)[1][1], mfp(x.smiles, fp_len, fp_rad).fp)
        original_encoding = make_encoding(original_molecule)"""

    return similarity, make_encoding