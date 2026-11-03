"""MolGenerator with a seeded molecule order.

MolGenerator shuffles the CSV with the global numpy random state, which is not
seeded when the dataset loads. The dataset cache keeps the splits as indices but
not the instances, which are rebuilt at every load, so every new process would
map the cached split indices (and the folds a cached oracle was trained on) to
different molecules. Here the shuffle uses its own seed.

Parameters: as MolGenerator, plus seed (default 0), hashed into the dataset name.
"""
import pandas as pd

from src.dataset.generators.mol_gen import MolGenerator, smile2graph


class SeededMolGenerator(MolGenerator):

    def check_configuration(self):
        super().check_configuration()
        self.local_config['parameters'].setdefault('seed', 0)

    def _read(self, path):
        data = pd.read_csv(path).sample(frac=1.0, random_state=self.local_config['parameters']['seed'])
        for smile, label in zip(data.smiles, data[self._data_label_name]):
            # Ids stay consecutive over the molecules RDKit can sanitize, as in MolGenerator.
            sanitized, g = smile2graph(len(self.dataset.instances), smile, label, self.dataset)
            if sanitized:
                self.dataset.instances.append(g)
