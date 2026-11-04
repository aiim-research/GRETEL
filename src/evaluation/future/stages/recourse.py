"""Independent Recourse stage using the Explanation's configured oracle."""
from src.evaluation.future.stages.probe_stage import ProbeStage


class Recourse(ProbeStage):
    probe = 'recourse'
