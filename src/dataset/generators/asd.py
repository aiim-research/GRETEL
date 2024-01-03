from os import listdir
from os.path import isfile, join

import numpy as np

from src.dataset.instances.graph import GraphInstance
from src.dataset.generators.base import Generator


class ASD(Generator):
    
    def init(self):
        base_path = self.local_config['parameters']['data_dir']
        self._td_file_path = join(base_path, 'td')
        self._asd_file_path = join(base_path, 'asd')
        self.generate_dataset()
    
    def generate_dataset(self):
        if not len(self.dataset.instances):
            for label, dir in enumerate([self._td_file_path, self._asd_file_path]):
                self.read(dir, label=label)
        #TODO pad node and edge features
        
    def read(self, path, label=0):
        files = [f for f in listdir(path) if isfile(join(path, f))]
        instance_id = len(self.dataset.instances)
        for filename in files:
            with open(join(path, filename), 'r') as f:
                instance = [[int(num) for num in line.split(' ')] for line in f]
                self.dataset.instances.append(GraphInstance(instance_id, label=label, data=np.array(instance, dtype=np.int32)))
                instance_id += 1