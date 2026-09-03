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
