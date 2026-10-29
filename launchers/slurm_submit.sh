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
# list of configs of the array is kept in output/slurm/<arrayjob>.list (task i = line i+1), and the
# scheduler's own output of each task (only what happens before the job opens its log) in output/slurm/.
# Resubmitting is safe: configs whose results already exist, and configs still queued or running in an
# earlier array, are skipped.
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
# Configs still queued or running in an earlier array of this user have no results yet, so
# pending_configs.py lists them again: leave them out, so that resubmitting never runs a config twice.
# Each array keeps its list as output/slurm/<arrayjob>.list, and task i runs line i+1.
{ squeue -h -r -u "$USER" -n gretel -o "%F %K" 2>/dev/null || true; } | while read -r J T; do
    if [[ $T =~ ^[0-9]+$ ]] && [ -f "output/slurm/$J.list" ]; then sed -n "$((T + 1))p" "output/slurm/$J.list"; fi
done | sort -u > "$TMP.active"
if [ -s "$TMP.active" ]; then
    echo "already queued or running, left out:"; sed 's/^/  /' "$TMP.active"
    grep -vxF -f "$TMP.active" "$TMP" > "$TMP.cut" || true
    mv "$TMP.cut" "$TMP"
fi
rm -f "$TMP.active"
if [ "$LIMIT" -gt 0 ]; then
    head -n "$LIMIT" "$TMP" > "$TMP.cut" && mv "$TMP.cut" "$TMP"
fi
N=$(wc -l < "$TMP")
if [ "$N" -eq 0 ]; then echo "nothing to submit"; rm -f "$TMP"; exit 0; fi

echo "submitting $N config(s) to partition $PART, at most $MAX at a time:"
sed 's/^/  /' "$TMP"
JOB=$(GRETEL_ENV=$ENV_NAME sbatch --parsable -p "$PART" --array=0-$((N - 1))%"$MAX" launchers/slurm_job.sh "$TMP")
cp "$TMP" "output/slurm/$JOB.list"
echo "array job $JOB  (list: output/slurm/$JOB.list, logs: output/logs/<dataset>/<generator>-<minimizer>/fold<f>_${JOB}_<task>.out)"
echo "watch it with: squeue -u \$USER   |   ls output/logs/*/*/*_${JOB}_*.out"
