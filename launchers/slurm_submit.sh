#!/bin/bash
# Submit the configs of one or more queue files that still have no results as one Slurm job array,
# with at most MAX tasks running at the same time. Run it from the repository root.
#
#   launchers/slurm_submit.sh -p <partition> [-m MAX] [-n LIMIT] [-e ENV] [-c] <queue.md> [<queue.md> ...]
#
#   -p  partition (required: check `sinfo -s` and `sacctmgr show assoc user=$USER format=partition`)
#   -m  maximum tasks running at once (default 10)
#   -n  submit only the first LIMIT pending configs (use -n 1 to test a single job first)
#   -e  conda environment activated inside each job (default GRTL)
#   -c  chain: one array per queue file, each starting once the previous one has ended (whatever its
#       tasks' exit status), instead of a single array for all of them
#
# Logs: output/logs/<dataset>/<generator>-<minimizer>/fold<f>_<arrayjob>_<task>.out, one per config. The
# list of configs of the array is kept in output/slurm/<arrayjob>.list (task i = line i+1), and the
# scheduler's own output of each task (only what happens before the job opens its log) in output/slurm/.
# Resubmitting is safe: configs whose results already exist, and configs still queued or running in an
# earlier array, are skipped.
set -euo pipefail

PART=""; MAX=10; LIMIT=0; ENV_NAME=${GRETEL_ENV:-GRTL}; CHAIN=0
while getopts "p:m:n:e:c" opt; do
    case $opt in
        p) PART=$OPTARG ;;
        m) MAX=$OPTARG ;;
        n) LIMIT=$OPTARG ;;
        e) ENV_NAME=$OPTARG ;;
        c) CHAIN=1 ;;
        *) sed -n '2,18p' "$0"; exit 1 ;;
    esac
done
shift $((OPTIND - 1))
if [ -z "$PART" ] || [ $# -eq 0 ]; then sed -n '2,18p' "$0"; exit 1; fi
mkdir -p output/logs output/slurm

# Configs still queued or running in an earlier array of this user have no results yet, so
# pending_configs.py lists them again: leave them out, so that resubmitting never runs a config twice.
# Each array keeps its list as output/slurm/<arrayjob>.list, and task i runs line i+1.
ACTIVE=output/slurm/active.$$
{ squeue -h -r -u "$USER" -n gretel -o "%F %K" 2>/dev/null || true; } | while read -r J T; do
    if [[ $T =~ ^[0-9]+$ ]] && [ -f "output/slurm/$J.list" ]; then sed -n "$((T + 1))p" "output/slurm/$J.list"; fi
done | sort -u > "$ACTIVE"
if [ -s "$ACTIVE" ]; then
    echo "already queued or running, left out: $(wc -l < "$ACTIVE") config(s)"
fi

# submit <dependency job or ""> <queue.md> [...]: prints the array job id (nothing if nothing is pending)
submit() {
    local after=$1; shift
    local tmp; tmp=output/slurm/$(date +%Y%m%d-%H%M%S)-$$-$RANDOM.list
    python scripts/pending_configs.py "$@" > "$tmp"
    grep -vxF -f "$ACTIVE" "$tmp" > "$tmp.cut" || true
    mv "$tmp.cut" "$tmp"
    if [ "$LIMIT" -gt 0 ]; then
        head -n "$LIMIT" "$tmp" > "$tmp.cut" && mv "$tmp.cut" "$tmp"
    fi
    local n; n=$(wc -l < "$tmp")
    if [ "$n" -eq 0 ]; then echo "nothing to submit from $*" >&2; rm -f "$tmp"; return; fi
    echo "submitting $n config(s) from $* to partition $PART, at most $MAX at a time${after:+, after job $after}:" >&2
    if [ "$n" -le 40 ]; then sed 's/^/  /' "$tmp" >&2; else head -3 "$tmp" | sed 's/^/  /' >&2; echo "  ... ($((n - 3)) more)" >&2; fi
    local job
    job=$(GRETEL_ENV=$ENV_NAME sbatch --parsable -p "$PART" ${after:+--dependency=afterany:$after} \
          --array=0-$((n - 1))%"$MAX" launchers/slurm_job.sh "$tmp")
    cp "$tmp" "output/slurm/$job.list"
    cat "$tmp" >> "$ACTIVE"               # a later stage of the chain never repeats these
    echo "array job $job  (list: output/slurm/$job.list)" >&2
    echo "$job"
}

JOBS=()
if [ "$CHAIN" -eq 1 ]; then
    prev=""
    for q in "$@"; do
        j=$(submit "$prev" "$q")
        if [ -n "$j" ]; then JOBS+=("$j"); prev=$j; fi
    done
else
    j=$(submit "" "$@")
    if [ -n "$j" ]; then JOBS+=("$j"); fi
fi
rm -f "$ACTIVE"
if [ ${#JOBS[@]} -gt 0 ]; then
    echo "logs: output/logs/<dataset>/<generator>-<minimizer>/fold<f>_<arrayjob>_<task>.out"
    echo "watch it with: squeue -u \$USER   (cancel everything: scancel ${JOBS[*]})"
fi
