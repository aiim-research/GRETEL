from src.explainer.future.utils.explainer_transform import ExplainerTransformMeta
from src.explainer.search.maccs import MACCSExplainer as MACCSExplainerOld


class MACCSExplainer(MACCSExplainerOld, metaclass=ExplainerTransformMeta):
    pass
