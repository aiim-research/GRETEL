from src.core.factory_base import Factory
from src.utils.cfg_utils import inject_dataset, inject_oracle, inject_explainer

class PlotterFactory(Factory):      
    
    def get_plotter(self, plotter_snippet, dataset, oracle, explainer):
        inject_dataset(plotter_snippet, dataset)
        inject_oracle(plotter_snippet, oracle)
        inject_explainer(plotter_snippet, explainer)
        return self._get_object(plotter_snippet)
            
    def get_plotters(self, config_list, dataset, oracle):
        return [self.get_plotter(obj, dataset, oracle) for obj in config_list]