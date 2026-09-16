from src.explainer.future.utils.explainer_transform import ExplainerTransformMeta
from src.explainer.legacy.search.p_rand import PRandExplainer as PRandExplainerOld


class PRandExplainer(PRandExplainerOld, metaclass=ExplainerTransformMeta):
    pass
