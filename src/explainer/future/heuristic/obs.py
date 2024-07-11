from src.explainer.future.utils.explainer_transform import ExplainerTransformMeta
from src.explainer.heuristic.obs import (
    ObliviousBidirectionalSearchExplainer as ObliviousBidirectionalSearchExplainerOld,
)


class ObliviousBidirectionalSearchExplainer(
    ObliviousBidirectionalSearchExplainerOld,
    metaclass=ExplainerTransformMeta,
):
    pass
