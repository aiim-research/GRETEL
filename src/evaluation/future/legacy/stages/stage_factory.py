from src.core.factory_base import Factory


class StageFactory(Factory):

    def get_stage(self, stage_snippet):
        """
        Creates a metric object out of the configuration snippet
        """
        return self._get_object(stage_snippet)
            
    def get_stages(self, config_list):
        return [self.get_stage(obj) for obj in config_list]