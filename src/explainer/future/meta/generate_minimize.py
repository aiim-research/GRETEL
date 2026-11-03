import os
import pickle
import time

from src.core.explainer_base import Explainer
from src.core.factory_base import get_class, get_instance_kvargs
from src.core.trainable_base import Trainable
from src.utils.cfg_utils import  inject_dataset, inject_oracle
import src.utils.explanations.functions as exp_tools
from src.future.explanation.local.graph_counterfactual import LocalGraphCounterfactualExplanation
from src.dataset.instances.graph import GraphInstance


class GenerateMinimize(Explainer):
    """This meta-explainer uses an explanation method to generate a first counterfactual and
    then a minimizer function to reduce the distance between the counterfactual and the original
    instance"""

    def check_configuration(self):
        super().check_configuration()

        if 'generator' not in self.local_config['parameters']:
            raise Exception('A generate-minimize method requires a generator')
        
        if 'minimizer' not in self.local_config['parameters']:
            raise Exception('A generate-minimize method requires a minimizer')

        # Inject the oracle and the dataset into the generator explainer
        inject_dataset(self.local_config['parameters']['generator'], self.dataset)
        inject_oracle(self.local_config['parameters']['generator'], self.oracle)
        # all the components should use the same fold
        self.local_config['parameters']['generator']['parameters']['fold_id'] = self.local_config['parameters']['fold_id']

        # Inject the oracle and the dataset into the minimizer
        inject_dataset(self.local_config['parameters']['minimizer'], self.dataset)
        inject_oracle(self.local_config['parameters']['minimizer'], self.oracle)
        # all the components should use the same fold
        self.local_config['parameters']['minimizer']['parameters']['fold_id'] = self.local_config['parameters']['fold_id']


    def init(self):
        super().init()
        self.logger = self.context.logger

        self.explanation_generator = get_instance_kvargs(self.local_config['parameters']['generator']['class'], 
                                                          {'context':self.context,'local_config': self.local_config['parameters']['generator']})
        
        self.explanation_minimizer = get_instance_kvargs(self.local_config['parameters']['minimizer']['class'], 
                                                          {'context':self.context,'local_config': self.local_config['parameters']['minimizer']})

        # Opt-in (read with .get, so configs without it keep their hash): a directory where the
        # generator's counterfactual is stored per (dataset, oracle, generator, fold, instance). The
        # generators are deterministic, so every minimizer that shares the generator reuses the same
        # starting counterfactual instead of recomputing it: faster, and the comparison between
        # minimizers is exactly paired. The generator's oracle calls are stored with it and added back
        # to the oracle's counter on a hit, so the OracleCalls metric keeps its meaning; the Runtime
        # metric of a hit no longer includes the generator.
        self.generator_cache = self.local_config['parameters'].get('generator_cache')


    def explain(self, instance):
        print("--------------------------------------------")

        initial_label = self.oracle.predict(instance)
        
        # Using the generator to obtain an initial explanation
        start_time = time.time()
        initial_explanation = self._generate(instance)
        
        initial_cf  = initial_explanation.counterfactual_instances[0]
        # initial_cf  = initial_explanation.counterfactual_instances[0]
        generator_runtime = time.time() - start_time # Getting the runtime of the generator
        initial_explanation._info['runtime'] = generator_runtime # Writing the runtime in the explanation
        initial_explanation._info['oracle_calls'] = self.oracle.get_calls_count() # Getting the number of oracle calls

        # Getting the predicted label of the initial explanation
        initial_cf_label = self.oracle.predict(initial_cf)
        # print("generated ctf -> " + str(initial_cf_label))

        # Let the option to the Minimizer
        # if initial_cf_label == initial_label:
        #    # the generator was not able to produce a counterfactual
        #    # so we can inmediately return, there is no point in minimizing
        #    self.logger.info(f'The generator could not generate a counterfactual for instance with id {str(instance.id)}')
        #    return initial_explanation
        # else:
        #     self.logger.info(f'The generator generated a counterfactual for instance with id {str(instance.id)}')
        
        # Try to minimize the distance between the counterfactual example and the original instance
        minimum_cf = self.explanation_minimizer.minimize(initial_explanation)
        
        # print("minimized ctf -> " + str(self.oracle.predict(minimum_cf)))

        minimal_explanation = LocalGraphCounterfactualExplanation(context=self.context,
                                                                    dataset=self.dataset,
                                                                    oracle=self.oracle,
                                                                    explainer=self,
                                                                    input_instance=instance,
                                                                    counterfactual_instances=[minimum_cf])
        
        minimal_explanation._info['generator_explanation'] = initial_explanation
        
        return minimal_explanation
    

    def _generator_cache_path(self, instance) -> str:
        return os.path.join(self.generator_cache, str(self.dataset.name), str(self.oracle.name),
                            str(self.explanation_generator.name), f"fold{self.local_config['parameters']['fold_id']}",
                            f"{instance.id}.pkl")

    def _generate(self, instance):
        """Run the generator, or replay its stored counterfactual when ``generator_cache`` is set."""
        if not self.generator_cache:
            return self.explanation_generator.explain(instance)
        path = self._generator_cache_path(instance)
        if os.path.exists(path):
            with open(path, "rb") as f:
                rec = pickle.load(f)
            cf = GraphInstance(id=rec["id"], label=rec["label"], data=rec["data"], node_features=rec["node_features"],
                               edge_features=rec["edge_features"], edge_weights=rec["edge_weights"],
                               graph_features=rec["graph_features"], dataset=self.dataset, directed=rec["directed"])
            self.oracle._call_counter += rec["oracle_calls"]   # the calls the generator made the first time
            self.logger.info(f"Generator output for instance {instance.id} replayed from cache "
                             f"({rec['oracle_calls']} generator oracle calls added back)")
            return LocalGraphCounterfactualExplanation(context=self.context, dataset=self.dataset, oracle=self.oracle,
                                                       explainer=self.explanation_generator, input_instance=instance,
                                                       counterfactual_instances=[cf])
        before = self.oracle.get_calls_count()
        explanation = self.explanation_generator.explain(instance)
        cf = explanation.counterfactual_instances[0]
        rec = {"id": cf.id, "label": cf.label, "data": cf.data, "node_features": cf.node_features,
               "edge_features": cf.edge_features, "edge_weights": cf.edge_weights,
               "graph_features": cf.graph_features, "directed": cf.directed,
               "oracle_calls": self.oracle.get_calls_count() - before}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "wb") as f:
            pickle.dump(rec, f)
        os.replace(tmp, path)                                    # atomic: parallel runs never read half a file
        return explanation

    @property
    def name(self):
        # Make the name of the generator_minimizer explainer show the names of the used generator and minimizer
        gen = get_class( self.local_config['parameters']['generator']['class'] ).__name__
        min = get_class( self.local_config['parameters']['minimizer']['class'] ).__name__
        alias = f'GenerateMinimize({gen}-{min})'
        return self.context.get_name(self,alias=alias)