# Retired snippets

`default_stages.json` declares the same eight evaluation stages, in the same
order, as the live `lab/config/snippets/default_pipeline.json` - but under the
`src.evaluation.stages.*Stage` class names from before the stages were folded
into `MainPipeline`. Those classes no longer exist, so the snippet cannot be
composed; `default_pipeline.json` replaces it.

Its only consumer is the neighbouring
`SE1-TCR-128-28_TCO_Ens[OBS+2xiRand+2xRSGG]-Bidirectional.jsonc`. It is kept
here rather than repaired because bringing it back means rewiring that config
to the `pipeline` key shape, a change to a retired experiment that nothing
verifies.
