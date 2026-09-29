import heapq
import random
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Iterable, List, Set, Tuple, Dict, Iterator, Optional


# =========================
# Fixed-size cache of full solutions
# =========================
class FixedSizeCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()

    def add(self, item: Iterable[int]):
        key = frozenset(item)
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            self.cache[key] = None
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)

    def contains(self, item: Iterable[int]) -> bool:
        return frozenset(item) in self.cache


# =========================
# Addition frontier using layered min-heap
# =========================
class AdditionFrontier:
    """
    Maintains a layered min-heap over the number line starting from the current base set.
    Key = (dist + soft_penalty, dist) so we pop closest first but respect cooldown.
    Never materializes the universe; validates with bounds and base-membership.
    Reused across proposals for the same baseline.
    """
    __slots__ = ("EPlus", "base", "add_penalty", "rcl_size", "rng", "heap", "seen")

    def __init__(self, EPlus: int, base: Set[int], add_penalty, rcl_size: int, rng: random.Random):
        self.EPlus = EPlus
        self.base = set(base)
        self.add_penalty = add_penalty
        self.rcl_size = rcl_size
        self.rng = rng
        self.heap: List[Tuple[float, int, int]] = []  # (key, dist, x)
        self.seen: Set[int] = set(self.base)
        self._seed_initial_neighbors()

    def pick_k(self, k: int) -> List[int]:
        """
        Pick k additions (not in base). Reuses/extends the heap across calls.
        Note: selecting items also grows the frontier; next calls will be faster/richer.
        Does NOT mutate self.base; caller decides to accept or not.
        """
        if k <= 0:
            return []
        picked: List[int] = []
        while len(picked) < k and self.heap:
            key, dist, x = heapq.heappop(self.heap)

            # Build a small RCL at same distance (actual proximity), sorted by key (penalty-aware)
            layer = [(key, dist, x)]
            while self.heap and self.heap[0][1] == dist and len(layer) < self.rcl_size:
                layer.append(heapq.heappop(self.heap))
            layer.sort(key=lambda t: t[0])
            _, d, px = self.rng.choice(layer)
            # push back unpicked
            for item in layer:
                if item[2] != px:
                    heapq.heappush(self.heap, item)

            # Validate and pick
            if 0 <= px < self.EPlus and px not in self.base and px not in picked:
                picked.append(px)
                # expand neighbors of picked
                self._push_neighbor(px - 1, d + 1)
                self._push_neighbor(px + 1, d + 1)

        # If the heap dries out (e.g., base is dense), do a small rejection-sample fallback
        if len(picked) < k:
            need = k - len(picked)
            picked.extend(self._random_fill(need, exclude=set(self.base).union(picked)))

        return picked

    # ---------- internal ----------
    def _seed_initial_neighbors(self):
        for s in self.base:
            self._push_neighbor(s - 1, 1)
            self._push_neighbor(s + 1, 1)

    def _push_neighbor(self, x: int, dist: int):
        if 0 <= x < self.EPlus and x not in self.seen:
            pen = self.add_penalty(x)  # soft bias against recent adds
            key = dist + pen
            heapq.heappush(self.heap, (key, dist, x))
            self.seen.add(x)

    def _random_fill(self, count: int, exclude: Set[int]) -> List[int]:
        out = set()
        attempts = 0
        max_attempts = max(10_000, 5 * count)
        while len(out) < count and attempts < max_attempts:
            x = self.rng.randrange(self.EPlus)
            if x not in exclude:
                out.add(x)
            attempts += 1
        return list(out)


# =========================
# Removal planner using nearest-neighbor gap + heap 
# =========================
class RemovalPlannerNN:
    """
    Builds proposals by simulating k removals from the BASE solution.
    Score(element) = nearest-neighbor gap (bigger = more isolated = prefer to remove)
    We subtract a soft cooldown penalty to *discourage re-removing* recent items.
    Uses a max-heap and updates only neighbors' scores when an item is (virtually) removed.
    Rebuilt when you change the baseline (accept()).
    """
    __slots__ = ("base_sorted", "remove_penalty", "rcl_size", "rng")

    def __init__(self, base: Set[int], rcl_size: int, remove_penalty, rng: random.Random):
        self.base_sorted = sorted(base)
        self.remove_penalty = remove_penalty
        self.rcl_size = rcl_size
        self.rng = rng

    def pick_k(self, k: int) -> List[int]:
        n = len(self.base_sorted)
        if k <= 0 or n == 0:
            return []
        k = min(k, n)

        # local mutable state
        alive = [True] * n
        left = [i-1 for i in range(n)]
        right = [i+1 for i in range(n)]
        right[-1] = -1

        def nn_gap(i: int) -> float:
            if not alive[i]:
                return -1.0
            lg = (self.base_sorted[i] - self.base_sorted[left[i]]) if left[i] != -1 else float("inf")
            rg = (self.base_sorted[right[i]] - self.base_sorted[i]) if right[i] != -1 else float("inf")
            if lg == float("inf") and rg == float("inf"):
                gap = float("inf")
            else:
                gap = min(lg, rg)
            pen = self.remove_penalty(self.base_sorted[i])  # soft cooldown
            return max(0.0, gap - pen)

        # heap items: (-score, version, idx) to emulate a max-heap
        heap: List[Tuple[float, int, int]] = []
        version = [0] * n
        for i in range(n):
            heapq.heappush(heap, (-nn_gap(i), version[i], i))

        def touch(i: int):
            """Recompute score for i and push a new heap item (lazy invalidation)."""
            if i == -1 or not alive[i]:
                return
            version[i] += 1
            heapq.heappush(heap, (-nn_gap(i), version[i], i))

        picked: List[int] = []
        remove_explore_prob = 0.03  # tiny exploration to avoid stalling

        while len(picked) < k and heap:
            explore = (self.rng.random() < remove_explore_prob)

            if not explore:
                # Build a small RCL from current valid top items
                layer = []
                while heap and len(layer) < self.rcl_size:
                    neg_score, ver, i = heapq.heappop(heap)
                    if not alive[i] or ver != version[i]:
                        continue  # stale
                    layer.append((neg_score, i))
                if not layer:
                    explore = True
                else:
                    # choose one from RCL uniformly; push back others
                    _, i = self.rng.choice(layer)
                    for item in layer:
                        if item[1] != i:
                            heapq.heappush(heap, (item[0], version[item[1]], item[1]))
            if explore:
                alive_idxs = [i for i in range(n) if alive[i]]
                if not alive_idxs:
                    break
                i = self.rng.choice(alive_idxs)

            # "remove" i in the simulation
            alive[i] = False
            picked.append(self.base_sorted[i])
            li, ri = left[i], right[i]
            if li != -1:
                right[li] = ri
                touch(li)
            if ri != -1:
                left[ri] = li
                touch(ri)

        return picked



# =========================
# SolutionExplorer with generators and accept()
# =========================
@dataclass
class SolutionExplorer:
    EPlus: int

    # Diversity/greediness knobs
    rcl_size_add: int = 8
    rcl_size_remove: int = 8

    # Soft action-specific cooldowns
    add_cooldown: int = 10
    remove_cooldown: int = 10
    add_penalty_weight: float = 0.75
    remove_penalty_weight: float = 0.75

    # Retry limit for skipping cached duplicates within one generator run
    max_skip_tries: int = 1000

    rng: random.Random = field(default_factory=random.Random)

    # Global memory
    solution_cache: FixedSizeCache = field(default_factory=lambda: FixedSizeCache(500000))
    _iter: int = 0
    _last_added_iter: Dict[int, int] = field(default_factory=dict)    # when an element was accepted as added
    _last_removed_iter: Dict[int, int] = field(default_factory=dict)  # when an element was accepted as removed

    # Per-baseline reusable state
    _add_frontiers: Dict[frozenset, AdditionFrontier] = field(default_factory=dict)
    _removal_planners: Dict[frozenset, RemovalPlannerNN] = field(default_factory=dict)

    # ------------------ acceptance hook ------------------
    def accept(self, new_solution: Set[int], prev_solution: Optional[Set[int]] = None):
        """
        Call this when a yielded candidate is GOOD and becomes your current baseline.
        - bumps iteration,
        - updates cooldown memories for elements actually added/removed,
        - records solution in cache,
        - rebinds per-baseline frontiers to the new baseline.
        """
        self._iter += 1
        if prev_solution is not None:
            added = set(new_solution) - set(prev_solution)
            removed = set(prev_solution) - set(new_solution)
            for x in added:
                self._last_added_iter[x] = self._iter
            for x in removed:
                self._last_removed_iter[x] = self._iter

        
        self._rebind_frontiers(new_baseline=new_solution)

    # ------------------ GENERATORS ------------------
    def gen_additions(self, base_solution: Set[int], ks: List[int], n: int) -> Iterator[Set[int]]:
        """
        Yield n candidate solutions made by adding k elements; k cycles over ks.
        Reuses a persistent frontier keyed by the BASE solution only.
        Does NOT mutate your base solution nor cooldowns; use accept() when you keep one.
        """
        assert ks, "ks must be a non-empty list"
        baseline_key = frozenset(base_solution)
        frontier = self._add_frontiers.get(baseline_key)
        if frontier is None:
            frontier = AdditionFrontier(
                EPlus=self.EPlus,
                base=set(base_solution),
                add_penalty=lambda x: self._add_penalty(x),
                rcl_size=self.rcl_size_add,
                rng=self.rng,
            )
            self._add_frontiers[baseline_key] = frontier

        yielded = set()
        i = 0
        tries = 0
        while len(yielded) < n and tries < self.max_skip_tries:
            k = ks[i % len(ks)]
            cand_added = frontier.pick_k(k)
            candidate = set(base_solution)
            for x in cand_added:
                if x not in candidate:
                    candidate.add(x)

            key = frozenset(candidate)
            if key not in yielded and not self.solution_cache.contains(candidate):
                yielded.add(key)
                self.solution_cache.add(candidate)
                yield candidate
            else:
                tries += 1
            i += 1

    def gen_removals(self, base_solution: Set[int], ks: List[int], n: int) -> Iterator[Set[int]]:
        """
        Yield n candidate solutions made by removing k elements; k cycles over ks.
        Uses a persistent nearest-neighbor-gap removal planner keyed by the BASE solution only.
        """
        assert ks, "ks must be a non-empty list"
        baseline_key = frozenset(base_solution)
        planner = self._removal_planners.get(baseline_key)
        if planner is None:
            planner = RemovalPlannerNN(
                base=set(base_solution),
                rcl_size=self.rcl_size_remove,
                remove_penalty=lambda x: self._remove_penalty(x),
                rng=self.rng,
            )
            self._removal_planners[baseline_key] = planner

        yielded = set()
        i = 0
        tries = 0
        while len(yielded) < n and tries < self.max_skip_tries:
            k = min(ks[i % len(ks)], len(base_solution))
            to_remove = planner.pick_k(k)
            candidate = set(base_solution)
            for x in to_remove:
                candidate.discard(x)

            key = frozenset(candidate)
            if key not in yielded and not self.solution_cache.contains(candidate):
                yielded.add(key)
                self.solution_cache.add(candidate)
                yield candidate
            else:
                tries += 1
            i += 1

    def gen_swaps(self, base_solution: Set[int], ks: List[int], n: int) -> Iterator[Set[int]]:
        """
        Yield n candidate solutions by removing k then adding k (k cycles over ks).
        Reuses the NN-gap removal planner; builds a short-lived add frontier from the post-removal state.
        """
        assert ks, "ks must be a non-empty list"
        baseline_key = frozenset(base_solution)
        planner = self._removal_planners.get(baseline_key)
        if planner is None:
            planner = RemovalPlannerNN(
                base=set(base_solution),
                rcl_size=self.rcl_size_remove,
                remove_penalty=lambda x: self._remove_penalty(x),
                rng=self.rng,
            )
            self._removal_planners[baseline_key] = planner

        yielded = set()
        i = 0
        tries = 0
        while len(yielded) < n and tries < self.max_skip_tries:
            k = min(ks[i % len(ks)], len(base_solution))

            # Remove k via planner
            to_remove = planner.pick_k(k)
            mid = set(base_solution)
            for x in to_remove:
                mid.discard(x)

            # Add k via a local frontier built from 'mid'
            add_frontier = AdditionFrontier(
                EPlus=self.EPlus,
                base=mid,
                add_penalty=lambda x: self._add_penalty(x),
                rcl_size=self.rcl_size_add,
                rng=self.rng,
            )
            to_add = add_frontier.pick_k(k)
            candidate = set(mid)
            for x in to_add:
                candidate.add(x)

            key = frozenset(candidate)
            if key not in yielded and not self.solution_cache.contains(candidate):
                yielded.add(key)
                yield candidate
                self.solution_cache.add(candidate)
            else:
                tries += 1
            i += 1

    # ------------------ INTERNAL: cooldown penalties ------------------
    def _add_penalty(self, x: int) -> float:
        """
        Soft penalty for re-adding an element that was added recently (even if later removed).
        Linear decay across add_cooldown iterations.
        """
        t = self._last_added_iter.get(x)
        if t is None or self.add_cooldown <= 0:
            return 0.0
        dt = max(0, self._iter - t)
        if dt >= self.add_cooldown:
            return 0.0
        return self.add_penalty_weight * (1.0 - dt / self.add_cooldown)

    def _remove_penalty(self, x: int) -> float:
        """
        Soft penalty to discourage removing an element again if it was removed recently.
        Active even if it's currently in the solution (meaning it was removed and re-added).
        """
        t = self._last_removed_iter.get(x)
        if t is None or self.remove_cooldown <= 0:
            return 0.0
        dt = max(0, self._iter - t)
        if dt >= self.remove_cooldown:
            return 0.0
        return self.remove_penalty_weight * (1.0 - dt / self.remove_cooldown)

    # ------------------ INTERNAL: rebind per-baseline state ------------------
    def _rebind_frontiers(self, new_baseline: Set[int]):
        """
        When you accept a candidate, rebase the stateful planners/frontiers to that solution.
        """
        self._add_frontiers.clear()
        self._removal_planners.clear()
        key = frozenset(new_baseline)
        self._add_frontiers[key] = AdditionFrontier(
            EPlus=self.EPlus,
            base=set(new_baseline),
            add_penalty=lambda x: self._add_penalty(x),
            rcl_size=self.rcl_size_add,
            rng=self.rng,
        )
        self._removal_planners[key] = RemovalPlannerNN(
            base=set(new_baseline),
            rcl_size=self.rcl_size_remove,
            remove_penalty=lambda x: self._remove_penalty(x),
            rng=self.rng,
        )


