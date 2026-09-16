# `src/utils/legacy`

The command-line pipeline GRETEL used before the config-driven runners in
`scripts/` and `tests/` existed. Kept because `docs/legacy/execution_pipeline.txt`
documents it and old result folders were produced with it:

```
python src/utils/legacy/generate_folds.py --config_file <cfg> --output_folder <dir>
sh launchers/experiments_kalifano.sh <dir>
python src/utils/legacy/generate_results.py --source_folder <results> --output_file results.csv
python src/utils/legacy/visualize.py --results_file results.csv --output_file image.png
```

`generate_results.py`, `visualize.py` and `datasetinspection.py` run their work
at import time, so they only work as scripts, never as modules.

Nothing under `src/` imports any of this. The current entry point is `main.py`
driven by a config, and the current batch runners are
`scripts/run_revision_queue.py` and `tests/run_experiments.py`.
