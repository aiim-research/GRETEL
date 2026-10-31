#!/bin/bash
# The selector-arm pattern of the local preliminary, on the Slurm server, in one command: queue A (the
# collections of folds 1 and 2 and bls_v1), then the pretraining of the datasets' checkpoints on a
# compute node once A has ended, then queue B (selector_v1) once the pretraining has ended. Run it from
# the repository root.
#
#   launchers/slurm_selector_arm.sh -p <partition> [-m MAX] <queueA.md> <queueB.md> <dataset> [<dataset> ...]
#
# <dataset> are the names of the collections and checkpoints (lab/output/selector_pretrain/<dataset>_collect-f*.pt).
set -euo pipefail
PART=""; MAX=10
while getopts "p:m:" opt; do
    case $opt in
        p) PART=$OPTARG ;;
        m) MAX=$OPTARG ;;
        *) sed -n '2,9p' "$0"; exit 1 ;;
    esac
done
shift $((OPTIND - 1))
if [ -z "$PART" ] || [ $# -lt 3 ]; then sed -n '2,9p' "$0"; exit 1; fi
QA=$1; QB=$2; shift 2
JA=$(launchers/slurm_submit.sh -p "$PART" -m "$MAX" "$QA" | sed -n 's/^JOBS //p')
JP=$(sbatch --parsable -p "$PART" ${JA:+--dependency=afterany:${JA// /:}} launchers/slurm_pretrain.sh "$@")
echo "pretraining job $JP${JA:+, after $JA} (log: output/slurm/pretrain_$JP.out)"
launchers/slurm_submit.sh -p "$PART" -m "$MAX" -a "$JP" "$QB"
