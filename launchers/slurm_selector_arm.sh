#!/bin/bash
# The selector-arm pattern of the local preliminary, on the Slurm server, in one command: queue A (the
# collections of folds 1 and 2 and bls_v1), then the pretraining of the datasets' checkpoints on a
# compute node once A has ended, then queue B (selector_v1) once the pretraining has ended. Run it from
# the repository root.
#
#   launchers/slurm_selector_arm.sh -p <partition> [-m MAX] [-a JOB[:JOB...]] <queueA.md> <queueB.md> <dataset> [<dataset> ...]
#
# -a: queue A starts only once those jobs have ended (warm jobs building caches or oracles).
#
# <dataset> are the names of the collections and checkpoints (lab/output/selector_pretrain/<dataset>_collect-f*.pt).
set -euo pipefail
PART=""; MAX=10; AFTER=""
while getopts "p:m:a:" opt; do
    case $opt in
        p) PART=$OPTARG ;;
        m) MAX=$OPTARG ;;
        a) AFTER=$OPTARG ;;
        *) sed -n '2,11p' "$0"; exit 1 ;;
    esac
done
shift $((OPTIND - 1))
if [ -z "$PART" ] || [ $# -lt 3 ]; then sed -n '2,11p' "$0"; exit 1; fi
QA=$1; QB=$2; shift 2
JA=$(launchers/slurm_submit.sh -p "$PART" -m "$MAX" ${AFTER:+-a "$AFTER"} "$QA" | sed -n 's/^JOBS //p')
DEP=${JA:-$AFTER}
JP=$(sbatch --parsable -p "$PART" ${DEP:+--dependency=afterany:${DEP// /:}} launchers/slurm_pretrain.sh "$@")
echo "pretraining job $JP${JA:+, after $JA} (log: output/slurm/pretrain_$JP.out)"
launchers/slurm_submit.sh -p "$PART" -m "$MAX" -a "$JP" "$QB"
