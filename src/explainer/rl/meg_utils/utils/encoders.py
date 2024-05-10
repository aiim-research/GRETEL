from abc import ABC, abstractmethod
import numpy as np
from src.explainer.rl.meg_utils.utils.molecular_data_instance import MolecularDataInstance
from src.dataset.instances.base import DataInstance
from src.explainer.rl.meg_utils.utils.fingerprints import Fingerprint
from rdkit import Chem
from rdkit.Chem import AllChem


class ActionEncoderAB(ABC):
    
    def __init__(self):
        self._name = 'meg_action_encoder'
        
    @abstractmethod
    def encode(self, action: DataInstance) -> np.array:
        pass
    
    def encode_actions(self, actions):
        return [self.encode(action) for action in actions]
    
class IDActionEncoder(ActionEncoderAB):
    
    def __init__(self):
        super(IDActionEncoder, self).__init__()
        self._name = 'meg_id_action_encoder'
        
    def encode(self, action: DataInstance) -> np.array:
        return action.to_numpy_array()
    
class MorganBitFingerprintActionEncoder(ActionEncoderAB):
    
    def __init__(self, fp_len=1024, fp_rad=2):
        super(MorganBitFingerprintActionEncoder, self).__init__()
        self._name = 'morgan_bit_fingerprint_action_encoder'
        print(fp_len, type(fp_len))
        print(fp_rad, type(fp_rad))
        self.fp_length = fp_len
        self.fp_radius = fp_rad
        
    def encode(self, action: DataInstance) -> np.array:
        assert isinstance(action, MolecularDataInstance)                
        #if action: 
        molecule = Chem.MolFromSmiles(action.smiles)
        if molecule is None:
            print(action.smiles)
            
        fp = AllChem.GetMorganFingerprintAsBitVect(molecule,
                                                   self.fp_radius,
                                                   self.fp_length)
        return Fingerprint(fp, self.fp_length).numpy()
        """else:
            raise ValueError(f'DataIntance with id={action.id}'\
                + f' does not have a valid SMILES representation')"""
    
class MorganCountFingerprintActionEncoder(ActionEncoderAB):
    
    def __init__(self, fp_length=1024, fp_radius=2):
        super(MorganCountFingerprintActionEncoder, self).__init__()
        self._name = 'meg_morgan_count_fingerprint_action_encoder'
        
        self.fp_length = fp_length
        self.fp_radius = fp_radius
        
    def encode(self, action: DataInstance) -> np.array:
        assert isinstance(action, MolecularDataInstance)
                
        fp = AllChem.GetHashedMorganFingerprint(action.molecule,
                                                self.fp_radius,
                                                self.fp_length,
                                                bitInfo=None)
        return Fingerprint(fp, self.fp_length).numpy()
    
class RDKitFingerprintActionEncoder(ActionEncoderAB):
    
    def __init__(self, fp_length=1024, fp_radius=2):
        super(RDKitFingerprintActionEncoder, self).__init__()
        self._name = 'meg_rdkit_fingerprint_action_encoder'
        
        self.fp_length = fp_length
        self.fp_radius = fp_radius
        
    def encode(self, action: DataInstance) -> np.array:
        assert isinstance(action, MolecularDataInstance)
                
        fp = Chem.RDKFingerprint(action.molecule,
                                 self.fp_radius,
                                 self.fp_length,
                                 bitInfo=None)
        return Fingerprint(fp, self.fp_length).numpy()
