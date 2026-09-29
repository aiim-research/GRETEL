# Seguro / default — un worker, CPU:
python scripts/run_revision_queue.py

# PC multi-core, sin GPU — 8 workers CPU:
python scripts/run_revision_queue.py --workers 8

# Una máquina con 2 GPUs, 4 workers (2 por GPU):
python scripts/run_revision_queue.py --workers 4 --gpu

# Fijar a GPUs específicas:
python scripts/run_revision_queue.py --workers 4 --gpu --gpus 0,3

# Resumir bajo nohup para sobrevivir desconexión SSH:
nohup python scripts/run_revision_queue.py --workers 4 --gpu > /tmp/queue.log 2>&1 &

## Careful: the startup sync rewrites this checklist

`run_revision_queue.py` syncs `REVISION_EXECUTION_ORDER.md` against
`lab/output/results/` on every startup, in both directions. Results are not
versioned, so running it (even with `--sync-only`) on a machine where they are
absent unchecks every entry and loses the record of what has been run.

Run it only where the results live. If you ran it elsewhere:

```
git checkout -- docs/revision/REVISION_EXECUTION_ORDER.md
```
