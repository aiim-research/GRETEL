
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

from src.oracle.tabulars.tabular_base import TabularOracle
from src.utils.cfg_utils import init_dflts_to_of

class SVMOracle(TabularOracle):

    def init(self):
        super().init()
        svm = LinearSVC(**self.local_config['parameters']['model']['parameters'])
        self.model = CalibratedClassifierCV(svm)

    def check_configuration(self):
        super().check_configuration()
        kls="sklearn.svm.LinearSVC"
        self.local_config['parameters']['model']['class'] = self.local_config['parameters']['model'].get('class',kls)
        init_dflts_to_of(self.local_config, 'model', kls) #Init the default accordingly to the nested Classifier
