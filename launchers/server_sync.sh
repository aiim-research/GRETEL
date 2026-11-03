#!/bin/bash
# Update the server's clone to the pushed branch. Run it from the repository root instead of `git pull`.
#
#   launchers/server_sync.sh [BACKUP_DIR]
#
# Caches and results are tracked in git, but a server that already ran experiments has its own untracked
# copies at the same paths, and a plain `git pull` refuses to overwrite them ("untracked working tree
# files would be overwritten by merge"). For every file the incoming commits add that already exists here
# untracked: if it is byte-identical (a result made here and committed from the local machine) it is
# removed, otherwise (a cache trained here, a result that differs) it is moved to BACKUP_DIR (default
# ~/gretel-untracked-backup/<date-time>) and listed. Then `git pull --ff-only`. Nothing tracked is touched.
set -euo pipefail

BACKUP=${1:-$HOME/gretel-untracked-backup/$(date +%Y%m%d-%H%M%S)}
git fetch
UP=$(git rev-parse --abbrev-ref --symbolic-full-name '@{u}')
moved=0
while IFS= read -r -d '' f; do
    if [ -e "$f" ] && ! git ls-files --error-unmatch -- "$f" >/dev/null 2>&1; then
        if git show "$UP:$f" | cmp -s - "$f"; then
            rm -- "$f"
        else
            mkdir -p "$BACKUP/$(dirname "$f")"
            mv -- "$f" "$BACKUP/$f"
            echo "moved aside (differs from the committed one): $f"
            moved=$((moved + 1))
        fi
    fi
done < <(git diff -z --name-only --diff-filter=A HEAD "$UP")
[ "$moved" -gt 0 ] && echo "$moved file(s) moved to $BACKUP"
git pull --ff-only
