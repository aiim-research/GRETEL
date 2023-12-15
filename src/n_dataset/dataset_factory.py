from src.core.factory_base import Factory
class DatasetFactory(Factory):

    def get_dataset(self, dataset_snippet):
        return self._get_object(dataset_snippet)
            
    def get_datasets(self, config_list):
        return [self.get_dataset(obj) for obj in config_list]