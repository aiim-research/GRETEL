from abc import ABC, abstractmethod

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from src.dataset.instances.base import DataInstance
from src.explainer.rl.meg_utils.utils.fingerprints import Fingerprint
from src.explainer.rl.meg_utils.utils.molecular_instance import MolecularInstance
from src.utils.context import Context


class ActionEncoderAB(ABC):
    def __init__(self, context: Context = None):
        self._name = "meg_action_encoder"
        self.context = context
        self.context.logger.info(f"Initialized ActionEncoderAB with name: {self._name}")

    @abstractmethod
    def encode(self, action: DataInstance) -> np.array:
        pass

    def encode_actions(self, actions):
        self.context.logger.info("Encoding multiple actions.")
        encoded_actions = [self.encode(action) for action in actions]
        self.context.logger.info(f"Encoded {len(encoded_actions)} actions.")
        return encoded_actions


class IDActionEncoder(ActionEncoderAB):
    def __init__(self, context: Context = None):
        super(IDActionEncoder, self).__init__(context=context)
        self._name = "meg_id_action_encoder"
        self.context.logger.info(f"Initialized IDActionEncoder with name: {self._name}")

    def encode(self, action: DataInstance) -> np.array:
        self.context.logger.info(f"Encoding action with ID: {action.id}")
        return action.data


class MorganBitFingerprintActionEncoder(ActionEncoderAB):
    def __init__(self, fp_len=1024, fp_rad=2, context: Context = None):
        super(MorganBitFingerprintActionEncoder, self).__init__(context=context)
        self._name = "morgan_bit_fingerprint_action_encoder"
        self.fp_length = fp_len
        self.fp_radius = fp_rad
        self.context.logger.info(f"Initialized MorganBitFingerprintActionEncoder with name: {self._name}, length: {fp_len}, radius: {fp_rad}")

    def encode(self, action: DataInstance) -> np.array:
        assert isinstance(action, MolecularInstance)
        self.context.logger.info(f"Encoding action with SMILES: {action.smiles}")
        molecule = Chem.MolFromSmiles(action.smiles)
        if molecule is None:
            self.context.logger.error(f"Invalid SMILES: {action.smiles}")
            raise ValueError(f"Invalid SMILES: {action.smiles}")

        fp = AllChem.GetMorganFingerprintAsBitVect(
            molecule, self.fp_radius, self.fp_length
        )
        encoded_fp = Fingerprint(fp, self.fp_length).numpy()
        self.context.logger.info(f"Encoded fingerprint of length: {len(encoded_fp)}")
        return encoded_fp


class MorganCountFingerprintActionEncoder(ActionEncoderAB):
    def __init__(self, fp_length=1024, fp_radius=2, context: Context = None):
        super(MorganCountFingerprintActionEncoder, self).__init__(context=context)
        self._name = "meg_morgan_count_fingerprint_action_encoder"
        self.fp_length = fp_length
        self.fp_radius = fp_radius
        self.context.logger.info(f"Initialized MorganCountFingerprintActionEncoder with name: {self._name}, length: {fp_length}, radius: {fp_radius}")

    def encode(self, action: DataInstance) -> np.array:
        assert isinstance(action, MolecularInstance)
        self.context.logger.info(f"Encoding action with molecule ID: {action.id}")

        fp = AllChem.GetHashedMorganFingerprint(
            action.molecule, self.fp_radius, self.fp_length, bitInfo=None
        )
        encoded_fp = Fingerprint(fp, self.fp_length).numpy()
        self.context.logger.info(f"Encoded fingerprint of length: {len(encoded_fp)}")
        return encoded_fp


class RDKitFingerprintActionEncoder(ActionEncoderAB):
    def __init__(self, fp_length=1024, fp_radius=2, context: Context = None):
        super(RDKitFingerprintActionEncoder, self).__init__(context=context)
        self._name = "meg_rdkit_fingerprint_action_encoder"
        self.fp_length = fp_length
        self.fp_radius = fp_radius
        self.context.logger.info(f"Initialized RDKitFingerprintActionEncoder with name: {self._name}, length: {fp_length}, radius: {fp_radius}")

    def encode(self, action: DataInstance) -> np.array:
        assert isinstance(action, MolecularInstance)
        self.context.logger.info(f"Encoding action with molecule ID: {action.id}")

        fp = Chem.RDKFingerprint(
            action.molecule, self.fp_radius, self.fp_length, bitInfo=None
        )
        encoded_fp = Fingerprint(fp, self.fp_length).numpy()
        self.context.logger.info(f"Encoded fingerprint of length: {len(encoded_fp)}")
        return encoded_fp
