"""Independent Reversal stage; no preceding Reconstruction result required."""
from src.evaluation.future.stages.probe_stage import ProbeStage


class Reversal(ProbeStage):
    probe = 'reversal'
