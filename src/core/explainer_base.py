from abc import ABCMeta, abstractmethod
from src.utils.context import Context
from src.core.configurable import Configurable
from src.utils.cfg_utils import retake_dataset, retake_oracle

class Explainer(Configurable, metaclass=ABCMeta):
    # Explainers whose models or priors need every graph of the dataset at one size (padded, or fixed by
    # construction) set this; check_fixed_size fails clearly when the graphs differ, as in protocol v2.
    requires_fixed_size = False
    
    def __init__(self, context: Context, local_config):
        self.dataset = retake_dataset(local_config)
        self.oracle = retake_oracle(local_config)
        super().__init__(context, local_config)
        

    @abstractmethod
    def explain(self, instance):
        pass

    def check_configuration(self):
        super().check_configuration()
        self.local_config['parameters']['fold_id'] =  self.local_config['parameters'].get('fold_id', -1)
        self.fold_id = self.local_config['parameters']['fold_id'] 
        check_fixed_size(self, self.dataset)


def check_fixed_size(component, dataset):
    """Fail when a component that needs one graph size gets a dataset whose graphs differ in size.

    A no-op for the padded datasets and the fixed-size ones (every v1 config); it stops the protocol v2
    configs (no padding) from running RSGG, CLEAR, CF2, the GAN explainers, DDBS, DFS or DBS on graphs of
    several sizes, where they would crash or, for DFS and DBS, silently skip graphs."""
    if not getattr(component, 'requires_fixed_size', False) or dataset is None:
        return
    sizes = {len(instance.data) for instance in dataset.instances}
    if len(sizes) > 1:
        raise ValueError(f'{type(component).__name__} needs every graph at one size (pad the dataset, or use a '
                         f'fixed-size dataset); this dataset has graphs of {len(sizes)} sizes')
