from src.explainer.future.utils.explainer_transform import ExplainerTransformMeta
# from src.explainer.generative.rsgg import RSGG as RSGGOld
from src.legacy.explainer.generative.rsgg import RSGG as RSGGOld #TODO Point towards the last version of RSGG once it is finished


class RSGG(RSGGOld, metaclass=ExplainerTransformMeta):
    pass
