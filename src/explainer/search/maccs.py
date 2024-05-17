import numpy as np
import copy
import exmol
import selfies as sf

from src.explanation.local.graph_counterfactual import LocalGraphCounterfactualExplanation
from src.dataset.generators import mol_gen
from src.dataset.instances.graph import GraphInstance
from src.dataset.dataset_base import Dataset
from src.core.explainer_base import Explainer
from src.core.oracle_base import Oracle
from src.core.trainable_base import Trainable
from src.core.factory_base import get_instance_kvargs
from src.utils.cfg_utils import get_dflts_to_of, init_dflts_to_of, inject_dataset, inject_oracle, retake_oracle, retake_dataset
from src.utils.metrics.ged import GraphEditDistanceMetric

class MACCSExplainer(Explainer):
    """Model Agnostic Counterfactual Compounds with STONED (MACCS)"""

    def check_configuration(self):
        super().check_configuration()

        dst_metric='src.evaluation.evaluation_metric_ged.GraphEditDistanceMetric'  

        #Check if the distance metric exist or build with its defaults:
        init_dflts_to_of(self.local_config, 'distance_metric', dst_metric)


    def init(self):
        super().init()
        self.distance_metric = get_instance_kvargs(self.local_config['parameters']['distance_metric']['class'], 
                                                    self.local_config['parameters']['distance_metric']['parameters'])
    

    def explain(self, instance):

        smiles = instance.graph_features['smile']
        clf = self._oracle_wrapper_creator(self.oracle, self.dataset)

        basic = exmol.get_basic_alphabet()
        stoned_kwargs = {"num_samples": 1500, "alphabet": basic, "max_mutations": 2}

        try:
            samples = exmol.sample_space(smiles, clf, batched=False, use_selfies=False,
            stoned_kwargs=stoned_kwargs, quiet=True)

            cfs = exmol.cf_explain(samples)
        except Exception as err:
            # In case there was a problem transforming the smile then return the input instance as explanation
            print('instance id:', str(instance.id))
            print(instance.graph_features['smile'])
            print(err.args)

            # Building the explanation instance
            exp = LocalGraphCounterfactualExplanation(explainer_class=self.name,
                                                      input_instance=instance,
                                                      counterfactual_instances=[copy.deepcopy(instance)]
                                                      )
            return exp

        # TODO: include all cf instances in the explanation. Check why it is starting from index 1
        if(len(cfs) > 1):
            min_cft_label = clf(cfs[1].smiles)
            _ , min_counterfactual = mol_gen.smile2graph(instance.id, 
                                                         cfs[1].smiles, 
                                                         min_cft_label, 
                                                         self.dataset)
            
            # Building the explanation instance
            exp = LocalGraphCounterfactualExplanation(explainer_class=self.name,
                                                      input_instance=instance,
                                                      counterfactual_instances=[min_counterfactual]
                                                      )
            return exp
        else:
            # Building the explanation instance
            exp = LocalGraphCounterfactualExplanation(explainer_class=self.name,
                                                      input_instance=instance,
                                                      counterfactual_instances=[copy.deepcopy(instance)]
                                                      )
            return exp

    def _oracle_wrapper_creator(self, oracle: Oracle, dataset: Dataset):
        """
        This function takes an oracle and return a function that takes the smiles
        of a molecule, transforms it into a DataInstance and returns the prediction 
        of the oracle for it
        """

        # The inner function uses the oracle, but does not receive it as a parameter
        def _oracle_wrapper(molecule_smiles):
            _ , inst = mol_gen.smile2graph(-1, molecule_smiles, 0, dataset)
            return oracle.predict(inst)

        return _oracle_wrapper
    
    