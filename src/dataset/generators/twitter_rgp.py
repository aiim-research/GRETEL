from .TUDataset import TUDataset

class Twitter_RGP(TUDataset):

    def init(self):
        TUDataset.init(self,"TWITTER-Real-Graph-Partial")