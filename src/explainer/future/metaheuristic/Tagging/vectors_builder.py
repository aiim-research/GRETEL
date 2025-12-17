import numpy as np
from typing import Callable, Literal, Optional
import math
from collections import deque

try:
    import networkx as nx
except ImportError:
    nx = None

Aggregator = Literal["sum", "product", "mean", "max", "min"]


class VectorsBuilder:
    n: int                    # number of nodes
    Eplus: int                # number of unordered pairs, n choose 2
    K: int                    # feature dimensions
    metrics: list[str]
    Matrix: np.ndarray        # (n, n) adjacency (assumed undirected, 0/1 or weights)
    X: np.ndarray             # (n, K) node-feature matrix

    def __init__(self, metrics: list[str], matrix: np.ndarray):
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("matrix must be a square (n, n) adjacency matrix")
        self.n = int(matrix.shape[0])
        self.Eplus = self.n * (self.n - 1) // 2
        self.K = len(metrics)
        self.metrics = metrics

        # NODE-WISE matrix: one row per node, one column per metric
        self.X = np.zeros((self.n, self.K), dtype=np.float32)

        self.Matrix = matrix.astype(np.float32, copy=False)

        for dim, metric in enumerate(metrics):
            self.fill(dim=dim, metric=metric)

    # -------- public API --------

    def fill(self, dim: int, metric: str, **kwargs) -> None:
        if not (0 <= dim < self.K):
            raise ValueError(f"dim must be in [0, {self.K-1}]")

        m = metric.lower()
        if m in {"degree", "degree_centrality", "deg_centrality"}:
            self.fill_dimension_degree_centrality(dim=dim, **kwargs)
        elif m in {"closeness", "closeness_centrality"}:
            self.fill_dimension_closeness_centrality(dim=dim, **kwargs)
        elif m in {"eigenvector", "eigenvector_centrality"}:
            self.fill_dimension_eigenvector_centrality(dim=dim, **kwargs)
        elif m in {"betweenness", "betweenness_centrality"}:
            self.fill_dimension_betweenness_centrality(dim=dim, **kwargs)
        elif m in {"katz", "katz_centrality"}:
            self.fill_dimension_katz_centrality(dim=dim, **kwargs)
        elif m in {"pagerank", "page_rank"}:
            self.fill_dimension_pagerank(dim=dim, **kwargs)
        elif m in {"component", "components", "component_id", "connected_components"}:
            self.fill_dimension_component_id(dim=dim, **kwargs)
        elif m in {"eccentricity", "ecc"}:
            self.fill_dimension_eccentricity_approx(dim=dim, **kwargs)
        elif m in {"coreness", "kcore", "k-core"}:
            self.fill_dimension_coreness(dim=dim, **kwargs)
        elif m in {"local_efficiency", "ego_efficiency"}:
            self.fill_dimension_local_efficiency(dim=dim, **kwargs)
        elif m in {"same_component", "same_component_flag"}:
            self.fill_dimension_same_component(dim=dim, **kwargs)
        elif m in {"core_buckets", "core_periphery"}:
            self.fill_dimension_core_buckets(dim=dim, **kwargs)
        elif m in {"local_clustering", "clustering"}:
            self.fill_dimension_local_clustering(dim=dim, **kwargs)
        elif m in {"triangles", "triangle_count"}:
            self.fill_dimension_triangle_count(dim=dim, **kwargs)
        elif m in {"common_neighbors", "cn"}:
            self.fill_dimension_common_neighbors(dim=dim, **kwargs)
        elif m in {"jaccard"}:
            self.fill_dimension_jaccard(dim=dim, **kwargs)
        elif m in {"adamic_adar", "aa"}:
            self.fill_dimension_adamic_adar(dim=dim, **kwargs)
        elif m in {"resource_allocation", "ra"}:
            self.fill_dimension_resource_allocation(dim=dim, **kwargs)
        elif m in {"avg_neighbor_degree", "average_neighbor_degree"}:
            self.fill_dimension_avg_neighbor_degree(dim=dim, **kwargs)
        elif m in {"ego_density", "egonet_density"}:
            self.fill_dimension_ego_density(dim=dim, **kwargs)

        else:
            raise ValueError(f"Unknown metric: {metric}")

    # -------- node-wise metrics (scores per node) --------

    def fill_dimension_degree_centrality(
        self,
        dim: int,
        aggregator: Aggregator = "sum",  # ignored in node-wise mode
        normalized: bool = True,
    ) -> None:
        deg = self.Matrix.sum(axis=1)
        scores = (deg / (self.n - 1)) if normalized else deg
        self._fill_from_scores(dim, scores.astype(np.float32), aggregator)

    def fill_dimension_closeness_centrality(
        self,
        dim: int,
        aggregator: Aggregator = "sum",  # ignored in node-wise mode
        wf_improved: bool = True,
    ) -> None:
        if nx is not None:
            G = self._to_nx_graph()
            cc = nx.closeness_centrality(G, wf_improved=wf_improved)
            scores = np.array([cc[i] for i in range(self.n)], dtype=np.float32)
        else:
            dist = self._all_pairs_shortest_path_lengths_unweighted()
            with np.errstate(divide="ignore", invalid="ignore"):
                reachable = np.isfinite(dist)
                s = (dist * reachable).sum(axis=1)
                r = reachable.sum(axis=1)
                if wf_improved:
                    scores = np.where(
                        s > 0,
                        (r - 1) / s * (r - 1) / (self.n - 1),
                        0.0,
                    )
                else:
                    scores = np.where(s > 0, (r - 1) / s, 0.0)
            scores = scores.astype(np.float32)

        self._fill_from_scores(dim, scores, aggregator)

    def fill_dimension_eigenvector_centrality(
        self,
        dim: int,
        aggregator: Aggregator = "sum",  # ignored in node-wise mode
        max_iter: int = 1000,
        tol: float = 1e-6,
    ) -> None:
        A = self.Matrix
        x = np.ones(self.n, dtype=np.float64) / np.sqrt(self.n)
        for _ in range(max_iter):
            x_new = A @ x
            norm = np.linalg.norm(x_new)
            if norm == 0.0:
                break
            x_new /= norm
            if np.linalg.norm(x_new - x) < tol:
                x = x_new
                break
            x = x_new
        if x.mean() < 0:
            x = -x
        scores = (x / x.max()) if x.max() > 0 else x
        self._fill_from_scores(dim, scores.astype(np.float32), aggregator)

    def fill_dimension_betweenness_centrality(
        self,
        dim: int,
        aggregator: Aggregator = "sum",  # ignored in node-wise mode
        normalized: bool = True,
        weight: Optional[str] = None,
    ) -> None:
        if nx is None:
            raise ImportError("betweenness_centrality requires networkx")
        G = self._to_nx_graph()
        bc = nx.betweenness_centrality(
            G,
            normalized=normalized,
            weight=None if weight is None else "weight",
        )
        scores = np.array([bc[i] for i in range(self.n)], dtype=np.float32)
        self._fill_from_scores(dim, scores, aggregator)

    def fill_dimension_katz_centrality(
        self,
        dim: int,
        aggregator: Aggregator = "sum",  # ignored in node-wise mode
        alpha: Optional[float] = None,
        beta: float = 1.0,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ) -> None:
        if nx is None:
            raise ImportError("katz_centrality requires networkx")
        G = self._to_nx_graph()
        if alpha is None:
            A = self.Matrix.astype(np.float64)
            x = np.ones(self.n) / np.sqrt(self.n)
            for _ in range(100):
                x = A @ x
                nrm = np.linalg.norm(x)
                if nrm == 0:
                    break
                x /= nrm
            lam = np.linalg.norm(A @ x)
            alpha = 0.85 / (lam + 1e-12) if lam > 0 else 0.1
        kc = nx.katz_centrality_numpy(G, alpha=alpha, beta=beta, normalized=True)
        scores = np.array([kc[i] for i in range(self.n)], dtype=np.float32)
        self._fill_from_scores(dim, scores, aggregator)

    def fill_dimension_pagerank(
        self,
        dim: int,
        aggregator: Aggregator = "sum",  # ignored in node-wise mode
        damping: float = 0.85,
        max_iter: int = 200,
        tol: float = 1.0e-06,
    ) -> None:
        if nx is None:
            raise ImportError("PageRank requires networkx")
        G = self._to_nx_graph()
        pr = nx.pagerank(G, alpha=damping, max_iter=max_iter, tol=tol)
        scores = np.array([pr[i] for i in range(self.n)], dtype=np.float32)
        self._fill_from_scores(dim, scores, aggregator)

    def fill_dimension_component_id(
        self,
        dim: int,
        aggregator: Aggregator = "mean",  # ignored in node-wise mode
        normalized: bool = True,
    ) -> None:
        comp_id, n_comp = self._connected_components_labels()
        scores = comp_id.astype(np.float32)
        if normalized and n_comp > 1:
            scores = scores / float(n_comp - 1)
        elif normalized:
            scores.fill(0.0)
        self._fill_from_scores(dim, scores, aggregator)

    def approximate_eccentricity_bounds(
        self,
        k: int = 8,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Approximate eccentricity and radius/diameter bounds using k BFS sweeps.

        Returns:
            ecc_approx: shape (n,), approximate eccentricity
            radius_bounds: (rad_lower, rad_upper)
            diameter_bounds: (diam_lower, diam_upper)
        """
        n = self.n
        if n == 0:
            return (
                np.zeros(0, dtype=np.float32),
                (0.0, 0.0),
                (0.0, 0.0),
            )

        nbr_lists = self._neighbors_list()
        _, deg = self._triangles_per_node()
        start = int(np.argmax(deg)) if n > 0 else 0

        lower = np.zeros(n, dtype=np.float64)
        upper = np.full(n, np.inf, dtype=np.float64)

        pivot = start
        pivots: list[int] = []

        for _ in range(max(1, k)):
            dist = self._bfs_distances_from(pivot, nbr_lists).astype(np.float64)
            finite = np.isfinite(dist)
            if not finite.any():
                break
            ecc_p = dist[finite].max()
            pivots.append(pivot)

            lower[finite] = np.maximum(lower[finite], dist[finite])
            upper[finite] = np.minimum(upper[finite], dist[finite] + ecc_p)

            next_pivot = int(np.argmax(np.where(finite, dist, -1.0)))
            if next_pivot == pivot:
                break
            pivot = next_pivot

        ecc_approx = np.where(
            np.isfinite(upper),
            0.5 * (lower + upper),
            lower,
        )

        finite_mask = np.isfinite(upper)
        if finite_mask.any():
            rad_lower = float(lower[finite_mask].min())
            rad_upper = float(upper[finite_mask].min())
            diam_lower = float(lower[finite_mask].max())
            diam_upper = float(upper[finite_mask].max())
        else:
            rad_lower = rad_upper = diam_lower = diam_upper = 0.0

        return ecc_approx.astype(np.float32), (rad_lower, rad_upper), (diam_lower, diam_upper)

    def fill_dimension_eccentricity_approx(
        self,
        dim: int,
        aggregator: Aggregator = "sum",  # ignored in node-wise mode
        k: int = 8,
        normalize: bool = True,
    ) -> None:
        ecc, _, _ = self.approximate_eccentricity_bounds(k=k)
        if normalize and ecc.size > 0:
            max_e = float(ecc.max())
            if max_e > 0:
                ecc = ecc / max_e
        self._fill_from_scores(dim, ecc.astype(np.float32), aggregator)

    def fill_dimension_coreness(
        self,
        dim: int,
        aggregator: Aggregator = "sum",  # ignored in node-wise mode
        normalize: bool = True,
    ) -> None:
        core = self._core_numbers().astype(np.float32)
        if normalize and core.size > 0:
            max_c = float(core.max())
            if max_c > 0:
                core = core / max_c
        self._fill_from_scores(dim, core, aggregator)

    def fill_dimension_local_efficiency(
        self,
        dim: int,
        aggregator: Aggregator = "mean",  # ignored in node-wise mode
    ) -> None:
        """
        Local efficiency E_loc(v) in ego network Γ(v) (node removed).
        """
        nbr_lists = self._neighbors_list()
        n = self.n
        eff = np.zeros(n, dtype=np.float32)

        for v in range(n):
            nbrs = nbr_lists[v]
            d = len(nbrs)
            if d < 2:
                eff[v] = 0.0
                continue

            idx_of = {int(u): i for i, u in enumerate(nbrs)}
            d_float = float(d)
            denom = d_float * (d_float - 1.0)
            sum_inv = 0.0

            for s_idx, s in enumerate(nbrs):
                dist = np.full(d, np.inf, dtype=np.float64)
                dist[s_idx] = 0.0
                q: deque[int] = deque([int(s)])
                q_idx: deque[int] = deque([s_idx])
                while q:
                    u = q.popleft()
                    u_idx = q_idx.popleft()
                    for w in nbr_lists[u]:
                        j = idx_of.get(int(w))
                        if j is None:
                            continue
                        if dist[j] == np.inf:
                            dist[j] = dist[u_idx] + 1.0
                            q.append(int(w))
                            q_idx.append(j)

                mask = np.isfinite(dist) & (dist > 0.0)
                if mask.any():
                    sum_inv += float((1.0 / dist[mask]).sum())

            eff[v] = float(sum_inv / denom)

        self._fill_from_scores(dim, eff.astype(np.float32), aggregator)

    def fill_dimension_local_clustering(
        self,
        dim: int,
        aggregator: Aggregator = "mean",  # ignored in node-wise mode
    ) -> None:
        tri, deg = self._triangles_per_node()
        n = self.n
        c = np.zeros(n, dtype=np.float32)
        for v in range(n):
            dv = int(deg[v])
            if dv < 2:
                c[v] = 0.0
            else:
                c[v] = float(2.0 * tri[v] / (dv * (dv - 1)))
        self._fill_from_scores(dim, c, aggregator)

    def fill_dimension_triangle_count(
        self,
        dim: int,
        aggregator: Aggregator = "sum",  # ignored in node-wise mode
        normalize: bool = True,
    ) -> None:
        tri, _ = self._triangles_per_node()
        tri = tri.astype(np.float32)
        if normalize and tri.size > 0:
            mx = float(tri.max())
            if mx > 0:
                tri = tri / mx
        self._fill_from_scores(dim, tri, aggregator)

    # -------- global metrics --------
    def fill_dimension_avg_neighbor_degree(self, dim: int, normalized: bool = True) -> None:
        A = (self.Matrix > 0).astype(np.float32, copy=False)
        deg = A.sum(axis=1)  # (n,)
        # sum of neighbor degrees: A @ deg
        sum_nbr_deg = A @ deg
        avg = np.zeros_like(deg, dtype=np.float32)
        mask = deg > 0
        avg[mask] = (sum_nbr_deg[mask] / deg[mask]).astype(np.float32)

        if normalized:
            # normalize by max possible degree (n-1) to keep scale stable
            avg = avg / max(1.0, float(self.n - 1))

        self._fill_from_scores(dim, avg.astype(np.float32), aggregator="sum")


    def fill_dimension_ego_density(self, dim: int, normalized: bool = True) -> None:
        # ego density: edges among neighbors / (deg choose 2)
        A = (self.Matrix > 0).astype(np.float32, copy=False)
        n = self.n
        scores = np.zeros(n, dtype=np.float32)

        for u in range(n):
            nbrs = np.flatnonzero(A[u] > 0)
            d = len(nbrs)
            if d < 2:
                scores[u] = 0.0
                continue
            sub = A[np.ix_(nbrs, nbrs)]
            # undirected: count edges among neighbors
            e = sub.sum() * 0.5
            denom = d * (d - 1) * 0.5
            scores[u] = float(e / denom) if denom > 0 else 0.0

        # already in [0,1] for simple graphs
        self._fill_from_scores(dim, scores.astype(np.float32), aggregator="sum")


    def degree_assortativity(self) -> float:
        """
        Global degree assortativity coefficient (Newman).
        """
        n = self.n
        if n == 0:
            return 0.0
        A = (self.Matrix > 0)
        deg = A.sum(axis=1).astype(np.float64)

        iu, ju = np.triu_indices(n, k=1)
        mask = A[iu, ju]
        u = iu[mask]
        v = ju[mask]
        m = float(len(u))
        if m == 0.0:
            return 0.0

        x = deg[u]
        y = deg[v]

        sum_x = x.sum()
        sum_y = y.sum()
        sum_x2 = (x ** 2).sum()
        sum_y2 = (y ** 2).sum()
        sum_xy = (x * y).sum()

        num = m * sum_xy - sum_x * sum_y
        den = m * 0.5 * (sum_x2 + sum_y2) - 0.25 * (sum_x + sum_y) ** 2
        return float(num / den) if den != 0.0 else 0.0

    # -------- helpers --------

    @staticmethod
    def _rank_with_ties(x: np.ndarray) -> np.ndarray:
        """
        Dense ranks with average for ties (Spearman-style).
        Returns ranks in [1, n].
        """
        x = np.asarray(x)
        n = x.size
        order = np.argsort(x)
        ranks = np.zeros(n, dtype=np.float64)

        i = 0
        while i < n:
            j = i + 1
            while j < n and x[order[j]] == x[order[i]]:
                j += 1
            avg_rank = 0.5 * (i + j - 1) + 1.0
            for k in range(i, j):
                ranks[order[k]] = avg_rank
            i = j
        return ranks

    @staticmethod
    def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
        """
        Spearman rank correlation ρ between x and y.
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape")

        rx = VectorsBuilder._rank_with_ties(x)
        ry = VectorsBuilder._rank_with_ties(y)

        rx_mean = rx.mean()
        ry_mean = ry.mean()
        num = ((rx - rx_mean) * (ry - ry_mean)).sum()
        den = math.sqrt(((rx - rx_mean) ** 2).sum() * ((ry - ry_mean) ** 2).sum())
        return float(num / den) if den != 0.0 else 0.0

    @staticmethod
    def kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
        """
        Naive O(n^2) Kendall τ (sufficient for moderate n).
        Counts concordant/discordant pairs, ignores ties.
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape")
        n = int(x.size)
        if n < 2:
            return 0.0

        concordant = 0
        discordant = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                dx = x[j] - x[i]
                dy = y[j] - y[i]
                if dx == 0 or dy == 0:
                    continue
                if dx * dy > 0:
                    concordant += 1
                else:
                    discordant += 1

        total = concordant + discordant
        if total == 0:
            return 0.0
        return float((concordant - discordant) / total)

    # -------- internal graph-structure helpers --------

    def _neighbors_list(self) -> list[np.ndarray]:
        """Return adjacency lists Γ(v) as numpy arrays of neighbors (undirected, >0 edges)."""
        A = (self.Matrix > 0)
        return [np.flatnonzero(A[v]) for v in range(self.n)]

    def _neighbors_sets_and_deg(self):
        """Return (neighbor_sets, degrees) for neighbor-intersection based metrics."""
        nbr_lists = self._neighbors_list()
        nbr_sets = [set(map(int, nbrs)) for nbrs in nbr_lists]
        deg = np.array([len(s) for s in nbr_sets], dtype=np.int32)
        return nbr_sets, deg

    def _bfs_distances_from(self, src: int, nbr_lists: list[np.ndarray]) -> np.ndarray:
        """Single-source unweighted BFS; returns distances (inf if unreachable)."""
        n = self.n
        dist = np.full(n, np.inf, dtype=np.float32)
        dist[src] = 0.0
        q: deque[int] = deque([src])
        while q:
            v = q.popleft()
            dv = dist[v]
            for u in nbr_lists[v]:
                if dist[u] == np.inf:
                    dist[u] = dv + 1.0
                    q.append(int(u))
        return dist

    def _connected_components_labels(self) -> tuple[np.ndarray, int]:
        """Return (comp_id per node, number_of_components) via BFS on undirected graph."""
        nbr_lists = self._neighbors_list()
        comp_id = -np.ones(self.n, dtype=np.int32)
        comp = 0
        for s in range(self.n):
            if comp_id[s] != -1:
                continue
            q: deque[int] = deque([s])
            comp_id[s] = comp
            while q:
                v = q.popleft()
                for u in nbr_lists[v]:
                    if comp_id[u] == -1:
                        comp_id[u] = comp
                        q.append(int(u))
            comp += 1
        return comp_id, comp

    def _core_numbers(self) -> np.ndarray:
        """
        k-core / coreness via Batagelj–Zaversnik O(n+m) algorithm.
        Returns core number c[v] for each node.
        """
        nbr_lists = self._neighbors_list()
        n = self.n
        deg = np.array([len(nbrs) for nbrs in nbr_lists], dtype=np.int32)
        if n == 0:
            return deg

        max_deg = int(deg.max(initial=0))
        bin_counts = np.zeros(max_deg + 1, dtype=np.int32)
        for d in deg:
            bin_counts[d] += 1

        start = 0
        for d in range(max_deg + 1):
            num = bin_counts[d]
            bin_counts[d] = start
            start += num

        vert = np.empty(n, dtype=np.int32)
        pos = np.empty(n, dtype=np.int32)

        next_index = bin_counts.copy()
        for v in range(n):
            d = deg[v]
            pos[v] = next_index[d]
            vert[next_index[d]] = v
            next_index[d] += 1

        for d in range(max_deg, 0, -1):
            bin_counts[d] = bin_counts[d - 1]
        bin_counts[0] = 0

        core = deg.copy()
        for i in range(n):
            v = int(vert[i])
            for u in nbr_lists[v]:
                u = int(u)
                if deg[u] > deg[v]:
                    du = deg[u]
                    pu = pos[u]
                    pw = bin_counts[du]
                    w = vert[pw]

                    if u != w:
                        vert[pu], vert[pw] = vert[pw], vert[pu]
                        pos[u], pos[w] = pw, pu

                    bin_counts[du] += 1
                    deg[u] -= 1
            core[v] = deg[v]

        return core.astype(np.int32)

    def _triangles_per_node(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Triangle counts per node and degrees, using edge-iterator + neighbor intersections.
        O(sum_{(u,v)∈E} min(deg u, deg v)).
        """
        nbr_sets, deg = self._neighbors_sets_and_deg()
        n = self.n
        A = (self.Matrix > 0)
        iu, ju = np.triu_indices(n, k=1)
        mask = A[iu, ju]
        u = iu[mask]
        v = ju[mask]
        tri = np.zeros(n, dtype=np.int64)

        for a, b in zip(u, v):
            a = int(a)
            b = int(b)
            if deg[a] > deg[b]:
                a, b = b, a
            common = nbr_sets[a].intersection(nbr_sets[b])
            c = len(common)
            if c == 0:
                continue
            tri[a] += c
            tri[b] += c
            for w in common:
                tri[w] += 1

        return tri, deg

    def _fill_from_scores(
        self,
        dim: int,
        scores: np.ndarray,
        aggregator: Aggregator = "sum",  # kept for compatibility, ignored
    ) -> None:
        """
        Given node scores (length n), fill X[:, dim] with those scores.
        Node-wise version: one score per node.
        """
        scores = np.asarray(scores, dtype=np.float32)
        if scores.shape != (self.n,):
            raise ValueError(f"scores must have shape ({self.n},), got {scores.shape}")
        self.X[:, dim] = scores

    def _to_nx_graph(self):
        """Undirected graph view of the adjacency matrix; weights kept if present."""
        if nx is None:
            raise ImportError("networkx not available")
        G = nx.Graph()
        G.add_nodes_from(range(self.n))
        A = self.Matrix
        iu, ju = np.triu_indices(self.n, k=1)
        w = A[iu, ju]
        nz = np.nonzero(w)[0]
        edges = [(int(iu[k]), int(ju[k]), float(w[k])) for k in nz]
        G.add_weighted_edges_from(edges)
        return G

    def _all_pairs_shortest_path_lengths_unweighted(self) -> np.ndarray:
        """BFS distances for an unweighted, undirected graph. inf if unreachable."""
        n = self.n
        A = (self.Matrix > 0).astype(np.uint8)
        dists = np.full((n, n), np.inf, dtype=np.float32)
        for s in range(n):
            dists[s, s] = 0.0
            q = [s]
            seen = np.zeros(n, dtype=bool)
            seen[s] = True
            while q:
                v = q.pop(0)
                nbrs = np.flatnonzero(A[v])
                for u in nbrs:
                    if not seen[u]:
                        seen[u] = True
                        dists[s, u] = dists[s, v] + 1.0
                        q.append(u)
        return dists

    @staticmethod
    def _pair_index(i: int, j: int, n: int) -> int:
        if not (0 <= i < j < n):
            raise ValueError("Require 0 <= i < j < n")
        return i * (2 * n - i - 1) // 2 + (j - i - 1)

    @staticmethod
    def _get_aggregator(name: Aggregator) -> Callable[[float, float], float]:
        if name == "sum":
            return lambda a, b: a + b
        if name == "product":
            return lambda a, b: a * b
        if name == "mean":
            return lambda a, b: 0.5 * (a + b)
        if name == "max":
            return lambda a, b: a if a >= b else b
        if name == "min":
            return lambda a, b: a if a <= b else b
        raise ValueError(f"Unknown aggregator: {name}")
