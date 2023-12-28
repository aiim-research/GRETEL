from src.explainer.rl.meg_utils.environments.molecule_env import Molecule
from src.explainer.rl.meg_utils.utils.similarity import get_similarity


class BBBPEnvironment(Molecule):
    
    def __init__(self,
                 oracle=None,
                 discount_factor=.9,
                 fp_len=1024,
                 fp_rad=2,
                 weight_sim=0.5,
                 similarity_measure='tanimoto',
                 **kwargs
                ):
        super(BBBPEnvironment, self).__init__(**kwargs)
        
        self.discount_factor = discount_factor
        self.oracle = oracle
        self.weight_sim = weight_sim
        
        self.similarity, self.make_encoding = get_similarity(similarity_measure,
                                                             oracle,
                                                             fp_len,
                                                             fp_rad)
        
    def reward(self):
        pred_score = self.oracle.predict_proba(self._state)
        
        sim_score = self.similarity(self.make_encoding(self._state).fp,
                                    self.make_encoding(self.init_instance).fp)
        
        reward = pred_score * (1- self.weight_sim) + sim_score * self.weight_sim
        
        return {
            'reward': reward * self.discount_factor,
            'reward_pred': pred_score,
            'reward_sim': sim_score,
        }