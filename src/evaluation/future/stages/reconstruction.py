"""Independent Reconstruction stage; scores ProbeNarratives' saved answers."""
from src.evaluation.future.stages.probe_stage import ProbeStage


class Reconstruction(ProbeStage):
    probe = 'reconstruction'
