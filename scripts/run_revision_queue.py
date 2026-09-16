#!/usr/bin/env python3
"""Queue runner for the REVISION batch (sequential or parallel).

Reads ``REVISION_EXECUTION_ORDER.md`` (markdown task list of config paths),
hands each unchecked entry to a worker thread, and flips the checkbox to
``[x]`` when the run completes and a result file shows up on disk.
Idempotent and restart-safe:

* On every startup the runner first **syncs** every unchecked entry
  against the filesystem - any config whose ``results_<fold>_<run>.json``
  already exists under ``lab/output/results/<scope>/`` is flipped to
  checked immediately. So previous successful runs (from earlier batches,
  manual runs, or interrupted runs that did finish writing) never re-run.
* Per-config subprocess isolation dodges the ``Context.__global`` trap
  (same pattern as ``tests/run_experiments.py``).
* Workers pop tasks from a thread-safe queue, so two workers never claim
  the same config. File writes and progress prints are also locked.
* If interrupted (Ctrl-C, kill, OS reboot, whatever), running the script
  again resumes exactly where it stopped - any config in flight at kill
  time stays unchecked, so it gets re-run; everything before it stays
  checked.

Outputs land where each config declares: ``lab/output/results/<scope>/``.
Per-config subprocess stdout/stderr is mirrored to
``lab/output/queue_logs/<scope>_fold<f>.log``.

Usage from the repo root::

    # Single-worker, CPU only (safe default):
    python scripts/run_revision_queue.py

    # Four workers in parallel, CPU only (multi-core box):
    python scripts/run_revision_queue.py --workers 4

    # GPU enabled; one worker per detected GPU, round-robin:
    python scripts/run_revision_queue.py --gpu --workers 2

    # Explicit GPU id list (round-robin), e.g. two workers pinned to GPUs 0,1:
    python scripts/run_revision_queue.py --gpu --gpus 0,1 --workers 2

    # Sync checkboxes with disk and exit (no runs):
    python scripts/run_revision_queue.py --sync-only

Run under ``nohup`` / ``tmux`` / ``screen`` to survive an SSH disconnect.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LIST_PATH = REPO / "REVISION_EXECUTION_ORDER.md"
RESULTS_ROOT = REPO / "lab" / "output" / "results"
LOG_DIR = REPO / "lab" / "output" / "queue_logs"
RUNNER_CHILD = REPO / "tests" / "run_experiments.py"

TASK_RE = re.compile(r"^- \[(?P<mark>[ x])\] (?P<path>.+)$")
FOLD_RE = re.compile(r"generate_minimize(\d+)\.jsonc$")
JSONC_COMMENT_BLOCK = re.compile(r"/\*.*?\*/", re.DOTALL)
JSONC_COMMENT_LINE = re.compile(r"//.*?\n")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _read_jsonc(path: Path) -> dict:
    txt = path.read_text()
    txt = JSONC_COMMENT_BLOCK.sub("", txt)
    txt = JSONC_COMMENT_LINE.sub("\n", txt)
    return json.loads(txt)


def _scope_of(rel_cfg: str) -> str:
    cfg = _read_jsonc(REPO / rel_cfg)
    return cfg["experiment"]["scope"]


def _fold_of(rel_cfg: str) -> int:
    m = FOLD_RE.search(rel_cfg)
    if not m:
        raise ValueError(f"could not extract fold id from {rel_cfg}")
    return int(m.group(1))


def _results_already(scope: str, fold: int, run_number: int) -> bool:
    scope_dir = RESULTS_ROOT / scope
    if not scope_dir.is_dir():
        return False
    target = f"results_{fold}_{run_number}.json"
    for _ in scope_dir.rglob(target):
        return True
    return False


def _parse_list(path: Path):
    """Return (raw_lines, tasks). ``tasks`` is a list of dicts."""
    lines = path.read_text().splitlines()
    tasks = []
    for line_no, line in enumerate(lines):
        m = TASK_RE.match(line)
        if not m:
            continue
        tasks.append({
            "line_no": line_no,
            "mark": m.group("mark"),
            "path": m.group("path").strip(),
        })
    return lines, tasks


def _save_list(path: Path, lines, tasks) -> None:
    """Write the list atomically, rebuilding only the task lines."""
    new_lines = list(lines)
    for t in tasks:
        new_lines[t["line_no"]] = f"- [{t['mark']}] {t['path']}"
    text = "\n".join(new_lines) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _sync_marks(tasks, run_number: int) -> tuple[int, int]:
    """Reconcile every checkbox with disk truth, both directions.

    A box is checked iff its ``results_<fold>_<run>.json`` exists. This makes
    the queue self-healing: deleting a scope's results (e.g. after a code fix
    invalidates them) automatically unchecks its boxes on the next startup, so
    they get re-run. Returns (n_checked, n_unchecked).
    """
    n_checked = n_unchecked = 0
    for t in tasks:
        try:
            scope = _scope_of(t["path"])
            fold = _fold_of(t["path"])
        except Exception:
            continue
        on_disk = _results_already(scope, fold, run_number)
        if on_disk and t["mark"] != "x":
            t["mark"] = "x"
            n_checked += 1
        elif not on_disk and t["mark"] == "x":
            t["mark"] = " "
            n_unchecked += 1
    return n_checked, n_unchecked


# ---------------------------------------------------------------------------
# GPU discovery
# ---------------------------------------------------------------------------


def _detect_gpu_ids() -> list[int]:
    """Best-effort: ask torch how many CUDA devices are visible. Returns
    ``[]`` when torch isn't installed or no CUDA is available."""
    try:
        import torch
    except ImportError:
        return []
    if not (hasattr(torch, "cuda") and torch.cuda.is_available()):
        return []
    return list(range(torch.cuda.device_count()))


def _resolve_gpu_assignment(use_gpu: bool, gpus_arg: str | None,
                            n_workers: int) -> list[int | None]:
    """Return a list of length ``n_workers`` mapping worker index ->
    GPU id (or ``None`` for CPU). Round-robins when ``len(gpus) < n_workers``.
    """
    if not use_gpu:
        return [None] * n_workers
    if gpus_arg is None:
        gpu_ids = _detect_gpu_ids()
        if not gpu_ids:
            print("WARNING: --gpu requested but no CUDA devices detected; "
                  "falling back to CPU for every worker.", file=sys.stderr)
            return [None] * n_workers
    else:
        try:
            gpu_ids = [int(x) for x in gpus_arg.split(",") if x.strip() != ""]
        except ValueError as e:
            print(f"--gpus must be a comma-separated list of integers: {e}",
                  file=sys.stderr)
            sys.exit(2)
        if not gpu_ids:
            print("--gpus parsed to an empty list; using CPU.", file=sys.stderr)
            return [None] * n_workers
    return [gpu_ids[i % len(gpu_ids)] for i in range(n_workers)]


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------


def _run_one(rel_cfg: str, scope: str, fold: int, run_number: int,
             timeout: int, gpu_id: int | None) -> tuple[bool, float, Path]:
    """Run one config in a subprocess. Returns (success, dur_sec, log_path).

    ``gpu_id`` controls device visibility for the child:
      * ``None``  -> ``CUDA_VISIBLE_DEVICES=""`` (force CPU)
      * ``int``   -> ``CUDA_VISIBLE_DEVICES="<id>"`` (pin to one GPU)
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{scope}_fold{fold}.log"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "" if gpu_id is None else str(gpu_id)

    t_start = time.time()
    with open(log_path, "w") as logf:
        logf.write(f"# config: {rel_cfg}\n# scope: {scope}\n# fold: {fold}\n")
        logf.write(f"# run_number: {run_number}\n"
                   f"# started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        logf.write(f"# device: {'CPU' if gpu_id is None else f'GPU#{gpu_id}'}\n")
        logf.write("# -----\n")
        logf.flush()
        try:
            proc = subprocess.run(
                [sys.executable, str(RUNNER_CHILD),
                 "--one", rel_cfg, "--run-number", str(run_number)],
                cwd=str(REPO),
                stdout=logf,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                env=env,
            )
            ok = (proc.returncode == 0)
        except subprocess.TimeoutExpired:
            logf.write(f"# TIMEOUT after {timeout}s\n")
            ok = False
        except Exception as e:
            logf.write(f"# RUNNER EXCEPTION: {type(e).__name__}: {e}\n")
            ok = False
    dur = time.time() - t_start

    # Trust the filesystem: success means the result JSON now exists.
    if ok and not _results_already(scope, fold, run_number):
        ok = False

    return ok, dur, log_path


# ---------------------------------------------------------------------------
# Worker pool
# ---------------------------------------------------------------------------


class _Pool:
    """Manual thread pool with per-worker GPU pinning and shared state.

    The task queue is the only source of truth for ``no overlap`` - workers
    pop atomically so two threads never claim the same task.
    """

    def __init__(self, list_path: Path, lines, tasks, run_number: int,
                 timeout: int, worker_gpus: list[int | None]):
        self.list_path = list_path
        self.lines = lines
        self.tasks = tasks
        self.run_number = run_number
        self.timeout = timeout
        self.worker_gpus = worker_gpus

        self.task_q: queue.Queue = queue.Queue()
        self.stop = threading.Event()
        self.file_lock = threading.Lock()
        self.print_lock = threading.Lock()
        self.counter_lock = threading.Lock()

        self.n_done = 0
        self.n_ok = 0
        self.n_fail = 0
        self.t0 = time.time()
        self.n_pending = 0

    def say(self, msg: str) -> None:
        with self.print_lock:
            print(msg, flush=True)

    def commit_mark(self, task) -> None:
        with self.file_lock:
            task["mark"] = "x"
            _save_list(self.list_path, self.lines, self.tasks)

    def run(self) -> tuple[int, int, int]:
        pending = [t for t in self.tasks if t["mark"] == " "]
        self.n_pending = len(pending)
        for t in pending:
            self.task_q.put(t)

        threads = []
        for w, gpu in enumerate(self.worker_gpus):
            th = threading.Thread(target=self._worker, args=(w, gpu),
                                  name=f"queue-w{w}", daemon=True)
            th.start()
            threads.append(th)

        # Wait until all queued tasks are claimed and processed.
        self.task_q.join()
        self.stop.set()
        # Wake workers if they're sleeping on a get().
        for _ in threads:
            self.task_q.put(None)
        for th in threads:
            th.join()

        return self.n_ok, self.n_fail, int(time.time() - self.t0)

    def _worker(self, worker_id: int, gpu_id: int | None) -> None:
        device_tag = "CPU" if gpu_id is None else f"GPU#{gpu_id}"
        self.say(f"[W{worker_id}] starting on {device_tag}")
        while not self.stop.is_set():
            try:
                task = self.task_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if task is None:
                self.task_q.task_done()
                return
            try:
                self._handle(worker_id, gpu_id, task)
            finally:
                self.task_q.task_done()

    def _handle(self, worker_id: int, gpu_id: int | None, task) -> None:
        rel = task["path"]
        try:
            scope = _scope_of(rel)
            fold = _fold_of(rel)
        except Exception as e:
            with self.counter_lock:
                self.n_done += 1
                self.n_fail += 1
                idx = self.n_done
            self.say(f"  [{idx}/{self.n_pending}] [W{worker_id}] "
                     f"PARSE_FAIL  {rel}: {type(e).__name__}: {e}")
            return

        # Last-chance disk check before we spend CPU/GPU on it.
        if _results_already(scope, fold, self.run_number):
            self.commit_mark(task)
            with self.counter_lock:
                self.n_done += 1
                self.n_ok += 1
                idx = self.n_done
            self.say(f"  [{idx}/{self.n_pending}] [W{worker_id}] "
                     f"SKIP (on disk)  {scope} fold={fold}")
            return

        with self.counter_lock:
            self.n_done += 1
            idx = self.n_done
        elapsed = time.time() - self.t0
        self.say(f"  [{idx}/{self.n_pending}] [W{worker_id}] RUN  "
                 f"{scope}/fold={fold}  (elapsed total={elapsed:.0f}s)")

        ok, dur, log_path = _run_one(rel, scope, fold, self.run_number,
                                     self.timeout, gpu_id)
        if ok:
            self.commit_mark(task)
            with self.counter_lock:
                self.n_ok += 1
            self.say(f"        [W{worker_id}] OK   in {dur:.0f}s  ({log_path.name})")
        else:
            with self.counter_lock:
                self.n_fail += 1
            self.say(f"        [W{worker_id}] FAIL in {dur:.0f}s  ({log_path}) "
                     "- left unchecked")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--timeout", type=int, default=7200,
                    help="per-config subprocess budget in seconds (default 2h)")
    ap.add_argument("--run-number", type=int, default=1,
                    help="written into results_<fold>_<run>.json (default 1)")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel workers (default 1). Each worker is its own "
                         "subprocess; the task queue guarantees no overlap.")
    ap.add_argument("--gpu", action="store_true",
                    help="allow workers to use CUDA. Without this flag, "
                         "CUDA_VISIBLE_DEVICES is forced to '' for every "
                         "subprocess (CPU only).")
    ap.add_argument("--gpus", default=None,
                    help="comma-separated GPU ids to use with --gpu, e.g. "
                         "'0,1'. Workers round-robin across this list. "
                         "Without it, all detected GPUs are used.")
    ap.add_argument("--sync-only", action="store_true",
                    help="flip checkboxes to match disk state and exit")
    ap.add_argument("--list", default=str(LIST_PATH),
                    help=f"path to the queue file (default {LIST_PATH.name})")
    args = ap.parse_args()

    if args.workers < 1:
        ap.error("--workers must be >= 1")

    list_path = Path(args.list)
    if not list_path.is_file():
        print(f"queue file not found: {list_path}", file=sys.stderr)
        return 2

    lines, tasks = _parse_list(list_path)
    n_total = len(tasks)
    if n_total == 0:
        print("queue is empty (no task lines parsed)")
        return 0

    print(f"== Startup sync against {RESULTS_ROOT} ==", flush=True)
    n_checked, n_unchecked = _sync_marks(tasks, args.run_number)
    _save_list(list_path, lines, tasks)
    done_at_start = sum(1 for t in tasks if t["mark"] == "x")
    pending = [t for t in tasks if t["mark"] == " "]
    print(f"  checked from disk: {n_checked}   unchecked (results gone): {n_unchecked}")
    print(f"  total: {n_total}, already done: {done_at_start}, "
          f"pending: {len(pending)}", flush=True)

    if args.sync_only:
        return 0
    if not pending:
        print("nothing to run.")
        return 0

    worker_gpus = _resolve_gpu_assignment(args.gpu, args.gpus, args.workers)
    device_summary = ", ".join(
        f"W{i}={'CPU' if g is None else f'GPU#{g}'}"
        for i, g in enumerate(worker_gpus)
    )
    print(f"== Draining queue ({len(pending)} configs, workers={args.workers}, "
          f"per-config timeout={args.timeout}s, run_number={args.run_number}) ==",
          flush=True)
    print(f"   devices: {device_summary}", flush=True)

    pool = _Pool(list_path, lines, tasks, args.run_number, args.timeout,
                 worker_gpus)
    n_ok, n_fail, wallclock = pool.run()
    print()
    print(f"== Queue drained. ok={n_ok}  fail={n_fail}  "
          f"wallclock={wallclock}s ==", flush=True)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
