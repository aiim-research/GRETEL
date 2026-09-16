from src.explainer.rl.meg_utils.environments.molecule_env import MoleculeEnvironment
from src.explainer.rl.meg_utils.utils.similarity import get_similarity
from src.utils.context import Context


class BBBPEnvironment(MoleculeEnvironment):
    def __init__(
        self,
        oracle=None,
        discount_factor=0.9,
        fp_len=1024,
        fp_rad=2,
        weight_sim=0.5,
        similarity_measure="tanimoto",
        context: Context = None,
        **kwargs,
    ):
        super(BBBPEnvironment, self).__init__(context=context, **kwargs)

        self.discount_factor = discount_factor
        self.oracle = oracle
        self.weight_sim = weight_sim

        self.similarity, self.make_encoding = get_similarity(
            similarity_measure, oracle, fp_len, fp_rad
        )
        self.pred_score_init = None

    def reward(self):
        class_index = self.init_instance.label
        if self.pred_score_init is None:
            self.pred_score_init = self.oracle.predict_proba(self.init_instance)
        pred_score_current = self.oracle.predict_proba(self._state)
        pred_score = pred_score_current[class_index] - self.pred_score_init[class_index]
        pred_score = pred_score.item()

        sim_score = self.similarity(
            self.make_encoding(self._state).fp,
            self.make_encoding(self.init_instance).fp,
        )

        reward = pred_score * (1 - self.weight_sim) + sim_score * self.weight_sim

        return {
            "reward": reward * self.discount_factor,
            "reward_pred": pred_score,
            "reward_sim": sim_score,
        }
