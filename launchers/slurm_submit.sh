#!/bin/bash
# Submit the configs of one or more queue files that still have no results as one Slurm job array,
# with at most MAX tasks running at the same time. Run it from the repository root.
#
#   launchers/slurm_submit.sh -p <partition> [-m MAX] [-n LIMIT] [-e ENV] <queue.md> [<queue.md> ...]
#
#   -p  partition (required: check `sinfo -s` and `sacctmgr show assoc user=$USER format=partition`)
#   -m  maximum tasks running at once (default 10)
#   -n  submit only the first LIMIT pending configs (use -n 1 to test a single job first)
#   -e  conda environment activated inside each job (default GRTL)
#
# Logs: output/logs/<dataset>/<generator>-<minimizer>/fold<f>_<arrayjob>_<task>.out, one per config. The
# list of configs of the array is kept in output/slurm/<date-time>.list (task i = line i+1), and the
# scheduler's own output of each task (only what happens before the job opens its log) in output/slurm/.
# Resubmitting is safe: configs whose results already exist are skipped.
set -euo pipefail

PART=""; MAX=10; LIMIT=0; ENV_NAME=${GRETEL_ENV:-GRTL}
while getopts "p:m:n:e:" opt; do
    case $opt in
        p) PART=$OPTARG ;;
        m) MAX=$OPTARG ;;
        n) LIMIT=$OPTARG ;;
        e) ENV_NAME=$OPTARG ;;
        *) sed -n '2,15p' "$0"; exit 1 ;;
    esac
done
shift $((OPTIND - 1))
if [ -z "$PART" ] || [ $# -eq 0 ]; then sed -n '2,15p' "$0"; exit 1; fi

mkdir -p output/logs output/slurm
TMP=output/slurm/$(date +%Y%m%d-%H%M%S).list
python scripts/pending_configs.py "$@" > "$TMP"
if [ "$LIMIT" -gt 0 ]; then
    head -n "$LIMIT" "$TMP" > "$TMP.cut" && mv "$TMP.cut" "$TMP"
fi
N=$(wc -l < "$TMP")
if [ "$N" -eq 0 ]; then echo "nothing to submit"; rm -f "$TMP"; exit 0; fi

echo "submitting $N config(s) to partition $PART, at most $MAX at a time:"
sed 's/^/  /' "$TMP"
JOB=$(GRETEL_ENV=$ENV_NAME sbatch --parsable -p "$PART" --array=0-$((N - 1))%"$MAX" launchers/slurm_job.sh "$TMP")
echo "array job $JOB  (list: $TMP, logs: output/logs/<dataset>/<generator>-<minimizer>/fold<f>_${JOB}_<task>.out)"
echo "watch it with: squeue -u \$USER   |   ls output/logs/*/*/*_${JOB}_*.out"
