from src.explainer.future.utils.explainer_transform import ExplainerTransformMeta
from src.explainer.search.dces import DCESExplainer as DCESExplainerOld


class DCESExplainer(DCESExplainerOld, metaclass=ExplainerTransformMeta):
    pass
