import numpy as np
import copy
import exmol
import selfies as sf

from src.dataset.instances.graph import GraphInstance
from src.dataset.dataset_base import Dataset
from src.core.explainer_base import Explainer
from src.core.oracle_base import Oracle
from src.core.trainable_base import Trainable

from src.core.factory_base import get_instance_kvargs
from src.utils.cfg_utils import get_dflts_to_of, init_dflts_to_of, inject_dataset, inject_oracle, retake_oracle, retake_dataset

from src.evaluation.evaluation_metric_ged import GraphEditDistanceMetric

class MACCSExplainer(Explainer):
    """Model Agnostic Counterfactual Compounds with STONED (MACCS)"""

    def check_configuration(self):
        super().check_configuration()

        dst_metric='src.evaluation.evaluation_metric_ged.GraphEditDistanceMetric'  

        #Check if the distance metric exist or build with its defaults:
        init_dflts_to_of(self.local_config, 'distance_metric', dst_metric)

        if not 'fold_id' in self.local_config['parameters']:
            self.local_config['parameters']['fold_id'] = -1


    def init(self):
        super().init()
        self.distance_metric = get_instance_kvargs(self.local_config['parameters']['distance_metric']['class'], 
                                                    self.local_config['parameters']['distance_metric']['parameters'])
        self.fold_id = self.local_config['parameters']['fold_id']
    

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
            print('instance id:', str(instance.id))
            print(instance.smiles)
            print(err.args)
            return instance

        if(len(cfs) > 1):
            # min_counterfactual = copy.deepcopy(instance)

            min_counterfactual = MolecularDataInstance(-1)
            min_counterfactual.smiles = cfs[1].smiles
            min_counterfactual.n_node_types = dataset.n_node_types
            min_counterfactual.max_n_nodes = dataset.max_n_nodes
            return min_counterfactual
        else:
            return instance

    def _oracle_wrapper_creator(self, oracle: Oracle, dataset: Dataset):
        """
        This function takes an oracle and return a function that takes the smiles of a molecule, transforms it into a DataInstance and returns the prediction of the oracle for it
        """

        # The inner function uses the oracle, but does not receive it as a parameter
        def _oracle_wrapper(molecule_smiles):
            inst = MolecularDataInstance(-1)
            inst.smiles = molecule_smiles
            inst.n_node_types = dataset.n_node_types
            inst.max_n_nodes = dataset.max_n_nodes

            return oracle.predict(inst)

        return _oracle_wrapper