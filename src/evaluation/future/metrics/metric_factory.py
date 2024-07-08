from src.core.factory_base import Factory
from src.utils.cfg_utils import inject_dataset, inject_oracle

class MetricFactory(Factory):

    def get_metric(self, metric_snippet):
        """
        Creates a metric object out of the configuration snippet
        """
        return self._get_object(metric_snippet)
            
    def get_metrics(self, config_list):
        return [self.get_metric(obj) for obj in config_list]