import os
import sys


sys.path.append(os.path.abspath(os.path.join('./')))
from src.utils.context import Context
from src.dataset.dataset_factory import DatasetFactory


if __name__ == "__main__":
    print(f"Generating context for: {sys.argv[1]}")
    context = Context.get_context(sys.argv[1])
    in_path= sys.argv[2]
    context.logger.info(f"Creating: {context.config_file} @ {in_path}" )
   
    context.factories['datasets'] = DatasetFactory(context)
    ds_snippet={
            "class": "src.dataset.dataset_base.Dataset",
            "parameters": {
                "generator": {
                    "class": "src.dataset.generators.treecycles_rand.TreeCyclesRand",
                    "parameters": {
                        "num_instances": 128,
                        "num_nodes_per_instance": 32,
                        "ratio_nodes_in_cycles": 0.2,
                        "seed": 8732887
                    }
                },
                "manipulators": [
                    {
                        "class": "src.dataset.manipulators.centralities.NodeCentrality",
                        "parameters": {}
                    },
                    {
                        "class": "src.dataset.manipulators.weights.EdgeWeights",
                        "parameters": {}
                    }
                ],
                "n_splits": 10,
                "shuffle": True
            }
        }
    
    dataset = context.factories['datasets'].get_dataset(ds_snippet)
