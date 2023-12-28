import numpy as np
import torch
from rdkit.DataStructs import ConvertToNumpyArray

class Fingerprint:
    def __init__(self, fingerprint, fp_length):
        self.fp = fingerprint
        self.fp_len = fp_length

    def is_valid(self):
        return self.fingerprint is None

    def numpy(self):
        np_ = np.zeros((1,))
        ConvertToNumpyArray(self.fp, np_)
        return np_

    def tensor(self):
        return torch.as_tensor(self.numpy())
    