from src.explainer.future.utils.explainer_transform import ExplainerTransformMeta
from src.explainer.rl.meg import MEGExplainer as MEGExplainerOld


class MEGExplainer(MEGExplainerOld, metaclass=ExplainerTransformMeta):
    pass
