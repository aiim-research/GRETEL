"""Prequential monitoring of the online edge selector (v2).

Answers two questions without spending oracle calls:

1. Is the selector training enough? Steps, examples, label balance, loss EMAs, gradient and update
   norms, dormant units, buffer size, head disagreement.
2. Does training improve the proposals? Prequential (predict-then-observe) scores: every tried move
   carries the model's score at proposal time, the oracle then provides the label, so log-loss, Brier
   and AUC of those pre-registered predictions are an honest online test. A skill score compares the
   model's log-loss with a base-rate predictor (skill > 0 means the model beats "always predict the
   average"). Per accepted move we also record the number of tries it took and the rank the accepted
   move had in the model's own ranking versus the expected rank under uniform sampling.

Granularity: a compact log line every ``log_every`` oracle calls inside an instance, and one CSV row
per instance (``<dir>/selector_<tag>_<dataset>.csv``) for learning curves across instances and folds.
"""

from __future__ import annotations

import csv
import math
import os
import time
from collections import defaultdict


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


class _Preq:
    """Prequential accumulator for one mode."""

    def __init__(self) -> None:
        self.n = 0
        self.sum_y = 0.0
        self.ll = 0.0
        self.brier = 0.0
        self.pairs: list[tuple[float, float]] = []      # (pred_prob, y) for AUC
        self.tries_to_success: list[int] = []
        self.rank_adv: list[float] = []                  # expected_rank - model_rank, normalised by M
        self.n_success = 0

    def add(self, pred_logit: float, y: float) -> None:
        p = min(1.0 - 1e-6, max(1e-6, _sigmoid(pred_logit)))
        self.n += 1
        self.sum_y += y
        self.ll += -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))
        self.brier += (p - y) ** 2
        if len(self.pairs) < 20000:
            self.pairs.append((p, y))

    def success(self, tries: int, model_rank: int | None, M: int | None) -> None:
        self.n_success += 1
        self.tries_to_success.append(int(tries))
        if model_rank is not None and M and M > 1:
            self.rank_adv.append(((M - 1) / 2.0 - model_rank) / float(M))

    def base_rate_ll(self) -> float:
        if self.n == 0:
            return float("nan")
        q = min(1.0 - 1e-6, max(1e-6, self.sum_y / self.n))
        return -(self.sum_y * math.log(q) + (self.n - self.sum_y) * math.log(1.0 - q)) / self.n

    def auc(self) -> float:
        pos = [p for p, y in self.pairs if y >= 0.5]
        neg = [p for p, y in self.pairs if y < 0.5]
        if not pos or not neg:
            return float("nan")
        # Mann-Whitney with ties = 0.5
        allv = sorted([(p, 1) for p in pos] + [(p, 0) for p in neg])
        ranks = {}
        i = 0
        while i < len(allv):
            j = i
            while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
                j += 1
            r = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks.setdefault(allv[k][0], r)
            i = j + 1
        rsum = sum(ranks[p] for p in pos)
        return (rsum - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))

    def summary(self) -> dict:
        n = max(1, self.n)
        ll = self.ll / n if self.n else float("nan")
        base = self.base_rate_ll()
        # skill is only meaningful when both outcomes were observed (base-rate log-loss not ~0)
        skill = (1.0 - ll / base) if (self.n >= 5 and base > 1e-2) else float("nan")
        tts = self.tries_to_success
        return {
            "n": self.n,
            "pos_rate": self.sum_y / n if self.n else float("nan"),
            "logloss": ll,
            "logloss_base": base,
            "skill": skill,
            "brier": self.brier / n if self.n else float("nan"),
            "auc": self.auc(),
            "successes": self.n_success,
            "tries_mean": (sum(tts) / len(tts)) if tts else float("nan"),
            "tries_median": (sorted(tts)[len(tts) // 2]) if tts else float("nan"),
            "rank_adv": (sum(self.rank_adv) / len(self.rank_adv)) if self.rank_adv else float("nan"),
        }


class SelectorMonitor:
    FIELDS = [
        "dataset", "fold", "instance_idx", "instance_id", "selector_mode", "instances_seen",
        "initial_size", "final_size", "oracle_calls", "wall_s",
        "steps_add", "steps_rem", "buf_add", "buf_rem",
        "rem_n", "rem_pos_rate", "rem_logloss", "rem_logloss_base", "rem_skill", "rem_auc", "rem_brier",
        "rem_successes", "rem_tries_mean", "rem_tries_median", "rem_rank_adv",
        "add_n", "add_pos_rate", "add_logloss", "add_logloss_base", "add_skill", "add_auc", "add_brier",
        "add_successes", "add_tries_mean", "add_tries_median", "add_rank_adv",
        "rem_bce_ema", "add_bce_ema", "rem_listwise_ema", "add_listwise_ema",
        "rem_grad_norm_ema", "add_grad_norm_ema", "rem_update_norm_ema", "add_update_norm_ema",
        "rem_head_std", "add_head_std", "rem_dormant", "add_dormant",
        "calls_remove", "calls_swap", "calls_add", "succ_remove", "succ_swap", "succ_add",
        "accepted_block_mean", "accepted_block_max", "retries", "rung_stats", "phase_shares", "method_success",
    ]

    def __init__(self, logger, csv_dir: str | None, dataset: str, tag: str, fold, selector_mode: str,
                 log_every: int = 200) -> None:
        self.logger = logger
        self.dataset = dataset
        self.fold = fold
        self.selector_mode = selector_mode
        self.log_every = int(log_every)
        self.csv_path = None
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
            self.csv_path = os.path.join(csv_dir, f"selector_{tag}_{dataset}.csv")
        self.instance_idx = -1
        self.reset_instance(None, 0)

    # ------------------------------------------------------------- lifecycle

    def reset_instance(self, instance_id, initial_size: int) -> None:
        self.instance_idx += 1
        self.instance_id = instance_id
        self.initial_size = int(initial_size)
        self.t0 = time.time()
        self.preq = {"remove": _Preq(), "add": _Preq()}
        self.tries = {"remove": 0, "add": 0}
        self.last_log_calls = 0

    # ------------------------------------------------------------- recording

    def observe(self, mode: str, pred_logit: float, y: float) -> None:
        """A move was tried: pre-registered prediction + observed label."""
        self.preq[mode].add(float(pred_logit), float(y))
        self.tries[mode] += 1

    def success(self, mode: str, model_rank: int | None = None, M: int | None = None) -> None:
        self.preq[mode].success(self.tries[mode], model_rank, M)
        self.tries[mode] = 0

    def sweep_reset(self, mode: str) -> None:
        self.tries[mode] = 0

    # --------------------------------------------------------------- logging

    def maybe_log(self, oracle_calls: int, best_size: int, selector) -> None:
        if oracle_calls - self.last_log_calls < self.log_every:
            return
        self.last_log_calls = oracle_calls
        self.logger.info("[monitor] " + self._line(oracle_calls, best_size, selector))

    def _line(self, oracle_calls: int, best_size: int, selector) -> str:
        parts = [f"calls={oracle_calls} best={best_size}"]
        st = selector.last_stats if selector is not None else {}
        for mode, short in (("remove", "rem"), ("add", "add")):
            s = self.preq[mode].summary()
            parts.append(
                f"{short}[n={s['n']} pos={s['pos_rate']:.3f} ll={s['logloss']:.3f}/{s['logloss_base']:.3f} "
                f"skill={s['skill']:+.3f} auc={s['auc']:.3f} succ={s['successes']} tries={s['tries_mean']:.1f} "
                f"rankadv={s['rank_adv']:+.3f} steps={selector.train_steps[mode] if selector else 0} "
                f"bce={st.get(f'{mode}_bce_ema', float('nan')):.3f} "
                f"grad={st.get(f'{mode}_grad_norm_ema', float('nan')):.3f} "
                f"hstd={st.get(f'{mode}_head_std_mean', float('nan')):.3f}]"
            )
        return " ".join(parts)

    def end_instance(self, final_size: int, oracle_calls: int, selector, extra: dict | None = None) -> dict:
        wall = time.time() - self.t0
        st = selector.last_stats if selector is not None else {}
        row = {
            "dataset": self.dataset, "fold": self.fold, "instance_idx": self.instance_idx,
            "instance_id": self.instance_id, "selector_mode": self.selector_mode,
            "instances_seen": getattr(selector, "instances_seen", -1),
            "initial_size": self.initial_size, "final_size": int(final_size),
            "oracle_calls": int(oracle_calls), "wall_s": round(wall, 2),
            "steps_add": selector.train_steps["add"] if selector else 0,
            "steps_rem": selector.train_steps["remove"] if selector else 0,
            "buf_add": len(selector.buffers["add"]) if selector else 0,
            "buf_rem": len(selector.buffers["remove"]) if selector else 0,
        }
        for mode, short in (("remove", "rem"), ("add", "add")):
            s = self.preq[mode].summary()
            for k in ("n", "pos_rate", "logloss", "logloss_base", "skill", "auc", "brier",
                      "successes", "tries_mean", "tries_median", "rank_adv"):
                row[f"{short}_{k}"] = s[k]
            row[f"{short}_bce_ema"] = st.get(f"{mode}_bce_ema", float("nan"))
            row[f"{short}_listwise_ema"] = st.get(f"{mode}_listwise_ema", float("nan"))
            row[f"{short}_grad_norm_ema"] = st.get(f"{mode}_grad_norm_ema", float("nan"))
            row[f"{short}_update_norm_ema"] = st.get(f"{mode}_update_norm_ema", float("nan"))
            row[f"{short}_head_std"] = st.get(f"{mode}_head_std_mean", float("nan"))
            row[f"{short}_dormant"] = st.get(f"{mode}_dormant_frac", float("nan"))
        if extra:
            row.update(extra)
        extra_txt = (" phases(calls rem/swap/add=%s/%s/%s succ=%s/%s/%s) block(mean=%.1f max=%s) retries=%s" % (
            extra.get("calls_remove"), extra.get("calls_swap"), extra.get("calls_add"), extra.get("succ_remove"),
            extra.get("succ_swap"), extra.get("succ_add"), extra.get("accepted_block_mean", float("nan")),
            extra.get("accepted_block_max"), extra.get("retries"))) if extra else ""
        if extra and extra.get("rung_stats"):
            extra_txt += f" rungs={extra['rung_stats']} shares={extra.get('phase_shares')} methods={extra.get('method_success')}"
        self.logger.info("[monitor][instance] " + self._line(oracle_calls, final_size, selector)
                         + f" wall={wall:.1f}s init={self.initial_size} final={final_size}" + extra_txt)
        if self.csv_path:
            new = not os.path.exists(self.csv_path)
            with open(self.csv_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self.FIELDS)
                if new:
                    w.writeheader()
                w.writerow({k: self._fmt(row.get(k)) for k in self.FIELDS})
        return row

    @staticmethod
    def _fmt(v):
        if isinstance(v, float):
            return "" if math.isnan(v) else f"{v:.5g}"
        return v
