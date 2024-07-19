from src.explainer.future.utils.explainer_transform import ExplainerTransformMeta
from src.explainer.generative.rsgg import RSGG as RSGGOld


class RSGG(RSGGOld, metaclass=ExplainerTransformMeta):
    pass
