from src.explainer.future.utils.explainer_transform import ExplainerTransformMeta
from src.explainer.heuristic.ddbs import (
    DataDrivenBidirectionalSearchExplainer as DataDrivenBidirectionalSearchExplainerOld,
)


class DataDrivenBidirectionalSearchExplainer(
    DataDrivenBidirectionalSearchExplainerOld,
    metaclass=ExplainerTransformMeta,
):
    pass
