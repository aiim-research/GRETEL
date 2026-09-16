# `launchers/`

Cluster submission scripts (`qsub` for SGE, `sbatch` for Slurm) from the
HPC phase of the project. They are site-specific: queue names, module loads
and the search directories are hardcoded for machines the project used, so
none of them runs unmodified elsewhere.

| Script | Submits |
|---|---|
| `launch.sh`, `m_launch.sh`, `r_launch.sh`, `glaunch.sh` | a single job wrapping `main.py` or `future_main.py` |
| `multi.sh`, `m_multi.sh`, `r_multi.sh`, `gmulti.sh`, `local_multi.sh` | one job per config found in a directory |
| `experiments.sh`, `experiments_kalifano.sh` | the sweep driven by `docs/legacy/execution_pipeline.txt` |
| `del_multi.sh` | cancels a submitted batch |
| `env_install.sh` | the pre-conda environment bootstrap |

For running the current batch on one machine, use
`scripts/run_revision_queue.py` instead: it parallelises over local workers or
GPUs, is restart-safe, and tracks progress in
`docs/revision/REVISION_EXECUTION_ORDER.md`.
