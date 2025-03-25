from src.explainer.future.utils.explainer_transform import ExplainerTransformMeta
# from src.explainer.generative.rsgg import RSGG as RSGGOld
# from src.legacy.explainer.generative.rsgg import RSGG as RSGGOld #TODO Point towards the last version of RSGG once it is finished
from src.legacy.explainer.rsgg_v2.generative.rsgg import RSGG as RSGGOld #TODO Point towards the last version of RSGG once it is finished
# from src.explainer.generative.rsgg import RSGG as RSGGOld


class RSGG(RSGGOld, metaclass=ExplainerTransformMeta):
    pass
