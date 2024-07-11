from src.explainer.future.utils.explainer_transform import ExplainerTransformMeta
from src.explainer.search.i_rand import IRandExplainer as IRandExplainerOld


class IRandExplainer(IRandExplainerOld, metaclass=ExplainerTransformMeta):
    pass
