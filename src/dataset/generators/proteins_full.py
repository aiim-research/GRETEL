from .TUDataset import TUDataset

class Proteins_Full(TUDataset):

    def init(self):
        TUDataset.init(self,"PROTEINS_full")