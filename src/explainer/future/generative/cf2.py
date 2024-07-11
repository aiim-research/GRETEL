from src.explainer.future.utils.explainer_transform import ExplainerTransformMeta
from src.explainer.generative.cf2 import CF2Explainer as CF2ExplainerOld


class CF2Explainer(CF2ExplainerOld, metaclass=ExplainerTransformMeta):
    pass
