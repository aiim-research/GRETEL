from src.dataset.data_instance_molecular import MolecularDataInstance
from src.evaluation.evaluation_metric_base import EvaluationMetric
from src.core.explainer_base import Explainer
from src.dataset.dataset_base import Dataset
from src.core.oracle_base import Oracle

import exmol
import selfies as sf

class MACCSExplainer(Explainer):
    """Model Agnostic Counterfactual Compounds with STONED (MACCS)"""

    def __init__(self, id, instance_distance_function : EvaluationMetric, config_dict=None) -> None:
        super().__init__(id, config_dict)
        self._gd = instance_distance_function
        self._name = 'MACSSExplainer'


    def explain(self, instance, oracle: Oracle, dataset: Dataset):

        smiles = instance.smiles
        clf = self._oracle_wrapper_creator(oracle, dataset)

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