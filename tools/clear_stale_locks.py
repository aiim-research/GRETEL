#!/usr/bin/env python3
"""Remove lock files left behind by runs that died.

Every cached component is written under a ``flufl.lock`` lock, so two runs
never train the same oracle at once. A run that is killed (Ctrl-C, a timeout,
an OOM) does not get to release its lock, and the claim file stays on disk.
Every later run of that component then waits for it.

It does not wait briefly. ``lock_release_tout`` is expressed in hours and the
configurations in this repository set it to 120, so one interrupted run can
block that dataset or oracle for five days. The symptom is a run that prints
nothing and eventually times out, which looks like a hang or a broken config
rather than a stale file.

This removes only locks whose owning process is **dead and on this machine**,
so it is safe to run while other experiments are in flight: a live run's lock
is left alone.

Usage::

    python tools/clear_stale_locks.py            # report only
    python tools/clear_stale_locks.py --apply
"""
from __future__ import annotations

import argparse
import os
import socket
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEARCH_ROOTS = ["lab/data", "lab/output", "data"]


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True          # exists, owned by someone else
    return True


def find_locks():
    """Yield (path, owner_host, owner_pid) for every lock artefact found.

    flufl.lock writes a claim file named ``<lock>.lck|<host>|<pid>|<nonce>``
    and points ``<lock>.lck`` at it.
    """
    for root in SEARCH_ROOTS:
        base = os.path.join(REPO, root)
        if not os.path.isdir(base):
            continue
        for dp, dn, fn in os.walk(base):
            for f in fn:
                if ".lck" not in f:
                    continue
                path = os.path.join(dp, f)
                parts = f.split("|")
                if len(parts) >= 3:
                    try:
                        yield path, parts[1], int(parts[2])
                        continue
                    except ValueError:
                        pass
                yield path, None, None          # the bare .lck link


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true",
                    help="delete the stale locks (default: report only)")
    args = ap.parse_args()

    host = socket.gethostname()
    stale, live, orphan_links = [], [], []
    for path, owner_host, pid in find_locks():
        if pid is None:
            orphan_links.append(path)
        elif owner_host != host:
            live.append((path, f"owned by {owner_host}, not this machine"))
        elif _alive(pid):
            live.append((path, f"pid {pid} is running"))
        else:
            stale.append((path, f"pid {pid} is gone"))

    # A bare .lck link is stale only once nothing claims it any more.
    claimed = {p.split("|")[0] for p, _ in live}
    orphan_links = [p for p in orphan_links if p not in claimed]

    for path, why in live:
        print(f"  keep   {os.path.relpath(path, REPO)}  ({why})")
    for path, why in stale:
        print(f"  stale  {os.path.relpath(path, REPO)}  ({why})")
    for path in orphan_links:
        print(f"  stale  {os.path.relpath(path, REPO)}  (claimed by nothing)")

    doomed = [p for p, _ in stale] + orphan_links
    print(f"\n{len(doomed)} stale, {len(live)} in use")
    if not doomed:
        return 0
    if not args.apply:
        print("re-run with --apply to delete the stale ones")
        return 0
    for p in doomed:
        try:
            os.unlink(p)
        except FileNotFoundError:
            pass
    print(f"deleted {len(doomed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
