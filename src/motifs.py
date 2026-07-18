"""Explicit, testable detectors for money-laundering graph motifs.

Every detector operates on a :class:`networkx.MultiDiGraph` whose edges carry:

* ``t``   -- transaction time as float epoch seconds,
* ``amount`` -- paid amount (float),
* ``edge_id`` -- stable integer id,
* ``is_laundering`` -- ground-truth flag (0/1), optional for pure detection.

The detectors are deliberately transparent (no learned parameters) so an AML
analyst can read the rule that fired. They also emit a per-edge feature vector
consumed by the LightGBM baseline and as node/edge attributes by the GNN.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import networkx as nx

from src.config import MotifWindows

HOUR_S: float = 3600.0

# Feature names produced by :func:`edge_features`, in a stable order.
FEATURE_NAMES: list[str] = [
    "fan_out_deg",
    "fan_in_deg",
    "in_cycle",
    "reciprocity",
    "amount_burst",
    "amount_log",
    "gather_scatter_hub",
    "scatter_gather_mid",
]


@dataclass
class MotifHit:
    """A single detected motif instance.

    Attributes:
        motif_type: One of ``fan_out``/``fan_in``/``cycle``/``gather_scatter``/
            ``scatter_gather``.
        edge_ids: Edge ids participating in the motif.
        center: The hub/pivot node (empty for cycles).

    """

    motif_type: str
    edge_ids: set[int] = field(default_factory=set)
    center: str = ""


def _out_edges(g: nx.MultiDiGraph, node: str) -> list[tuple[float, str, int]]:
    """Return ``(t, dst, edge_id)`` for a node's out-edges, sorted by time."""
    rows = [(float(d["t"]), v, int(d["edge_id"])) for _, v, d in g.out_edges(node, data=True)]
    return sorted(rows)


def _in_edges(g: nx.MultiDiGraph, node: str) -> list[tuple[float, str, int]]:
    """Return ``(t, src, edge_id)`` for a node's in-edges, sorted by time."""
    rows = [(float(d["t"]), u, int(d["edge_id"])) for u, _, d in g.in_edges(node, data=True)]
    return sorted(rows)


def _window_distinct_counts(rows: list[tuple[float, str, int]], window_s: float) -> dict[int, int]:
    """For each edge, count distinct counterparties within a forward window.

    Given time-sorted ``(t, party, edge_id)`` rows, returns ``edge_id -> number
    of distinct parties`` observed in ``[t, t + window_s]``. Two-pointer sweep,
    O(n log n) per node.
    """
    result: dict[int, int] = {}
    n = len(rows)
    right = 0
    counts: dict[str, int] = defaultdict(int)
    for left in range(n):
        if right < left:
            right = left
            counts.clear()
        # Extend the window as far as the time bound allows.
        while right < n and rows[right][0] <= rows[left][0] + window_s:
            counts[rows[right][1]] += 1
            right += 1
        result[rows[left][2]] = len([c for c in counts.values() if c > 0])
        # Drop the left endpoint before it leaves the window next iteration.
        party = rows[left][1]
        counts[party] -= 1
        if counts[party] <= 0:
            counts.pop(party, None)
    return result


# --- Detectors -------------------------------------------------------------
def detect_fan_out(g: nx.MultiDiGraph, window_h: float, min_degree: int) -> list[MotifHit]:
    """1 -> N: a node paying many distinct accounts inside one time window."""
    hits: list[MotifHit] = []
    window_s = window_h * HOUR_S
    for node in g.nodes:
        rows = _out_edges(g, node)
        if len(rows) < min_degree:
            continue
        counts = _window_distinct_counts(rows, window_s)
        peak = max(counts.values(), default=0)
        if peak >= min_degree:
            eids = {eid for _, _, eid in rows}
            hits.append(MotifHit("fan_out", eids, node))
    return hits


def detect_fan_in(g: nx.MultiDiGraph, window_h: float, min_degree: int) -> list[MotifHit]:
    """N -> 1: a node receiving from many distinct accounts in one window."""
    hits: list[MotifHit] = []
    window_s = window_h * HOUR_S
    for node in g.nodes:
        rows = _in_edges(g, node)
        if len(rows) < min_degree:
            continue
        counts = _window_distinct_counts(rows, window_s)
        peak = max(counts.values(), default=0)
        if peak >= min_degree:
            eids = {eid for _, _, eid in rows}
            hits.append(MotifHit("fan_in", eids, node))
    return hits


def detect_cycles(
    g: nx.MultiDiGraph, max_len: int = 6, window_h: float = 168.0, max_cycles: int = 10000
) -> list[MotifHit]:
    """A -> ... -> A with strictly increasing timestamps (money returns home).

    Uses :func:`networkx.simple_cycles` on the simple directed projection with a
    length bound, then verifies a temporally-increasing edge ordering exists
    around the cycle within ``window_h``.
    """
    simple = nx.DiGraph()
    # Drop self-loops (A->A "deposits"): they are yielded by simple_cycles as
    # length-1 cycles and would otherwise exhaust the max_cycles budget before
    # any genuine multi-account cycle is reached.
    simple.add_edges_from((u, v) for u, v in g.edges() if u != v)
    window_s = window_h * HOUR_S
    hits: list[MotifHit] = []
    seen = 0
    for cycle in nx.simple_cycles(simple, length_bound=max_len):
        seen += 1
        if seen > max_cycles:
            break
        if len(cycle) < 2:
            continue
        # A temporal cycle is monotonic in time from exactly one starting point;
        # try every rotation so detection does not depend on simple_cycles' order.
        for r in range(len(cycle)):
            rotated = cycle[r:] + cycle[:r]
            edge_ids = _temporal_cycle_edges(g, rotated + [rotated[0]], window_s)
            if edge_ids is not None:
                hits.append(MotifHit("cycle", edge_ids, ""))
                break
    return hits


def _temporal_cycle_edges(
    g: nx.MultiDiGraph, ordered: list[str], window_s: float
) -> set[int] | None:
    """Return edge ids realising ``ordered`` with increasing t, else ``None``."""
    chosen: set[int] = set()
    last_t = float("-inf")
    start_t: float | None = None
    for u, v in zip(ordered[:-1], ordered[1:]):
        if not g.has_edge(u, v):
            return None
        candidates = sorted((float(d["t"]), int(d["edge_id"])) for _, d in g[u][v].items())
        pick = next(((t, e) for t, e in candidates if t > last_t), None)
        if pick is None:
            return None
        t, eid = pick
        if start_t is None:
            start_t = t
        if t - start_t > window_s:
            return None
        last_t = t
        chosen.add(eid)
    return chosen


def detect_gather_scatter(g: nx.MultiDiGraph, window_h: float, min_degree: int) -> list[MotifHit]:
    """N -> 1 -> M: a hub gathers from many, then scatters to many."""
    hits: list[MotifHit] = []
    window_s = window_h * HOUR_S
    for node in g.nodes:
        ins = _in_edges(g, node)
        outs = _out_edges(g, node)
        n_in = len({u for _, u, _ in ins})
        n_out = len({v for _, v, _ in outs})
        if n_in < min_degree or n_out < min_degree:
            continue
        last_in = max((t for t, _, _ in ins), default=None)
        first_out = min((t for t, _, _ in outs), default=None)
        # gather must (mostly) precede scatter within the window
        if last_in is None or first_out is None:
            continue
        if first_out >= min((t for t, _, _ in ins), default=first_out) and (
            first_out - min((t for t, _, _ in ins), default=first_out) <= window_s
        ):
            eids = {e for _, _, e in ins} | {e for _, _, e in outs}
            hits.append(MotifHit("gather_scatter", eids, node))
    return hits


def detect_scatter_gather(g: nx.MultiDiGraph, window_h: float, min_degree: int) -> list[MotifHit]:
    """1 -> M -> 1: a source scatters through intermediaries to one collector."""
    hits: list[MotifHit] = []
    window_s = window_h * HOUR_S
    for src in g.nodes:
        succ = set(g.successors(src))
        if len(succ) < min_degree:
            continue
        # Candidate collectors are successors-of-successors reached from many mids.
        collectors: dict[str, set[str]] = defaultdict(set)
        for mid in succ:
            for coll in g.successors(mid):
                if coll not in (src, mid):
                    collectors[coll].add(mid)
        for coll, mids in collectors.items():
            if len(mids) < min_degree:
                continue
            eids = _scatter_gather_edges(g, src, coll, mids, window_s)
            if eids is not None:
                hits.append(MotifHit("scatter_gather", eids, src))
    return hits


def _scatter_gather_edges(
    g: nx.MultiDiGraph, src: str, coll: str, mids: set[str], window_s: float
) -> set[int] | None:
    """Collect edge ids for a temporally-consistent scatter-gather instance."""
    chosen: set[int] = set()
    start_t = float("inf")
    end_t = float("-inf")
    for mid in mids:
        scatter = min(
            ((float(d["t"]), int(d["edge_id"])) for _, d in g[src][mid].items()), default=None
        )
        gather = min(
            ((float(d["t"]), int(d["edge_id"])) for _, d in g[mid][coll].items()), default=None
        )
        if scatter is None or gather is None or gather[0] < scatter[0]:
            continue
        chosen.update((scatter[1], gather[1]))
        start_t = min(start_t, scatter[0])
        end_t = max(end_t, gather[0])
    if len(chosen) < 2 * len(mids) or end_t - start_t > window_s:
        return None
    return chosen


# --- Aggregation + per-edge features --------------------------------------
def detect_all(
    g: nx.MultiDiGraph, windows: MotifWindows, min_degree: int
) -> dict[str, list[MotifHit]]:
    """Run every detector and group the hits by motif type."""
    return {
        "fan_out": detect_fan_out(g, windows.fan_window_h, min_degree),
        "fan_in": detect_fan_in(g, windows.fan_window_h, min_degree),
        "cycle": detect_cycles(g, window_h=windows.cycle_window_h),
        "gather_scatter": detect_gather_scatter(g, windows.gather_scatter_window_h, min_degree),
        "scatter_gather": detect_scatter_gather(g, windows.gather_scatter_window_h, min_degree),
    }


def edge_features(
    g: nx.MultiDiGraph, windows: MotifWindows, min_degree: int
) -> tuple[dict[int, list[float]], dict[str, list[MotifHit]]]:
    """Compute the per-edge feature vector used by every downstream model.

    Returns:
        A tuple ``(features, hits)`` where ``features`` maps ``edge_id`` to a
        list aligned with :data:`FEATURE_NAMES`, and ``hits`` is the output of
        :func:`detect_all` (reused for evaluation and tracing).

    """
    import math

    hits = detect_all(g, windows, min_degree)

    # fan-out/fan-in membership is carried by the window-degree features below;
    # only the structural motifs need explicit membership flags.
    cycle_edges = _union_edges(hits["cycle"])
    gs_edges = _union_edges(hits["gather_scatter"])
    sg_edges = _union_edges(hits["scatter_gather"])

    fan_out_counts = _per_edge_window_degree(g, windows.fan_window_h, outgoing=True)
    fan_in_counts = _per_edge_window_degree(g, windows.fan_window_h, outgoing=False)
    node_amount_stats = _node_amount_stats(g)

    features: dict[int, list[float]] = {}
    for u, v, d in g.edges(data=True):
        eid = int(d["edge_id"])
        amount = float(d.get("amount", 0.0))
        mean, std = node_amount_stats.get(u, (amount, 1.0))
        burst = (amount - mean) / (std + 1e-6)
        reciprocity = 1.0 if g.has_edge(v, u) else 0.0
        features[eid] = [
            float(fan_out_counts.get(eid, 0)),
            float(fan_in_counts.get(eid, 0)),
            1.0 if eid in cycle_edges else 0.0,
            reciprocity,
            burst,
            math.log1p(max(amount, 0.0)),
            1.0 if (eid in gs_edges) else 0.0,
            1.0 if (eid in sg_edges) else 0.0,
        ]
    return features, hits


def _union_edges(hits: list[MotifHit]) -> set[int]:
    """Union of all edge ids across a list of hits."""
    out: set[int] = set()
    for h in hits:
        out |= h.edge_ids
    return out


def _per_edge_window_degree(g: nx.MultiDiGraph, window_h: float, outgoing: bool) -> dict[int, int]:
    """Distinct-counterparty count in the forward window, per edge."""
    window_s = window_h * HOUR_S
    counts: dict[int, int] = {}
    for node in g.nodes:
        rows = _out_edges(g, node) if outgoing else _in_edges(g, node)
        counts.update(_window_distinct_counts(rows, window_s))
    return counts


def _node_amount_stats(g: nx.MultiDiGraph) -> dict[str, tuple[float, float]]:
    """Mean/std of each node's outgoing amounts (for the amount-burst feature)."""
    sums: dict[str, float] = defaultdict(float)
    sqs: dict[str, float] = defaultdict(float)
    n: dict[str, int] = defaultdict(int)
    for u, _, d in g.edges(data=True):
        a = float(d.get("amount", 0.0))
        sums[u] += a
        sqs[u] += a * a
        n[u] += 1
    stats: dict[str, tuple[float, float]] = {}
    for node in n:
        mean = sums[node] / n[node]
        var = max(sqs[node] / n[node] - mean * mean, 0.0)
        stats[node] = (mean, var**0.5)
    return stats
