from src.explainer.future.utils.explainer_transform import ExplainerTransformMeta
from src.explainer.legacy.generative.gcountergan import GCounteRGAN as GCounteRGANOld


class GCounteRGAN(GCounteRGANOld, metaclass=ExplainerTransformMeta):
    pass
