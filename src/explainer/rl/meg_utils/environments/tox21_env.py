import torch
from torch.nn import functional as F

from src.explainer.rl.meg_utils.environments.molecule_env import (
    MoleculeEnvironment,
)
from src.explainer.rl.meg_utils.utils.molecules import mol_to_tox21_pyg
from src.explainer.rl.meg_utils.utils.similarity import get_similarity


class CF_Tox21(MoleculeEnvironment):
    def __init__(
        self,
        model_to_explain,
        original_molecule,
        discount_factor,
        fp_len,
        fp_rad,
        weight_sim=0.5,
        similarity_measure="tanimoto",
        **kwargs,
    ):
        super(CF_Tox21, self).__init__(**kwargs)

        self.class_to_optimize = 1 - original_molecule.y.item()
        self.discount_factor = discount_factor
        self.model_to_explain = model_to_explain
        self.weight_sim = weight_sim

        self.similarity, self.make_encoding = get_similarity(
            similarity_measure, model_to_explain, original_molecule, fp_len, fp_rad
        )

    def reward(self):
        data = mol_to_tox21_pyg(self._state.molecule)

        out, (_, encoding) = self.model_to_explain(data.x, data.edge_index)
        out = F.softmax(out, dim=-1).squeeze().detach()

        sim_score = self.similarity(self.make_encoding(data), self.original_encoding)
        pred_score = out[self.class_to_optimize].item()
        pred_class = torch.argmax(out).item()

        reward = pred_score * (1 - self.weight_sim) + sim_score * self.weight_sim

        return {
            "pyg": data,
            "reward": reward * self.discount_factor,
            "reward_pred": pred_score,
            "reward_sim": sim_score,
            "encoding": encoding.numpy(),
            "prediction": {
                "type": "bin_classification",
                "output": out.numpy().tolist(),
                "for_explanation": pred_class,
                "class": pred_class,
            },
        }
