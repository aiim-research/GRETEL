from src.dataset.manipulators.base import BaseManipulator


class FeatureNames(BaseManipulator):
    """Name the node-feature columns; the feature values do not change.

    Loaders such as TUDataset call the columns attribute_0, attribute_1, ...
    The `names` parameter lists one name per column, in column order, so
    explanations can refer to, e.g., the chemical symbol of a one-hot column.
    """

    def check_configuration(self):
        super().check_configuration()
        names = self.local_config['parameters']['names']
        if len(set(names)) != len(names):
            raise ValueError('Feature names must be distinct')

    def process(self):
        names = self.local_config['parameters']['names']
        for instance in self.dataset.instances:
            if instance.node_features.shape[1] != len(names):
                raise ValueError(f'{len(names)} names for {instance.node_features.shape[1]} feature columns')
        self.dataset.node_features_map = {name: column for column, name in enumerate(names)}

    def process_instance(self, instance):
        pass
