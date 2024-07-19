from src.explainer.future.utils.explainer_transform import ExplainerTransformMeta
from src.explainer.generative.clear import CLEARExplainer as CLEARExplainerOld


class CLEARExplainer(CLEARExplainerOld, metaclass=ExplainerTransformMeta):
    pass
