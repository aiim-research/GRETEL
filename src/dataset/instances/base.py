from typing import Optional
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.dataset.dataset_base import Dataset


class DataInstance:
    def __init__(self, id, label, data, dataset: Optional["Dataset"] = None):
        self.id = id
        self.data = data
        self.label = label # TODO: Refactoring to have a one-hot encoding of labels!
        self._dataset = dataset


    @property
    def dataset(self):
        return self._dataset
