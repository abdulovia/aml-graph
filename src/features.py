"""Assemble the per-edge feature matrix shared by the baseline and the GNN."""

from __future__ import annotations

import networkx as nx
import numpy as np

from src.config import RunConfig
from src.motifs import FEATURE_NAMES, MotifHit, edge_features


class EdgeDataset:
    """Aligned arrays describing every edge for supervised learning.

    Attributes:
        edge_ids: Int array of edge ids, order matches ``X``/``y``/``times``.
        X: Feature matrix ``(n_edges, len(FEATURE_NAMES))``.
        y: Binary illicit labels.
        times: Edge timestamps (epoch seconds) for the temporal split.
        amounts: Paid amount per edge (for the demo/API and narratives).
        src/dst: Node id per edge (for the GNN edge index).
        feature_names: Names of the columns of ``X``.
        motif_edge_map: motif_type -> set of edge ids (for recall-by-motif).
        hits: Raw detector output (for tracing/visualisation).

    """

    def __init__(self, g: nx.MultiDiGraph, cfg: RunConfig) -> None:
        """Build the dataset by running the motif feature extractor on ``g``."""
        feats, hits = edge_features(g, cfg.windows, cfg.min_fan_degree)
        rows: list[tuple[int, str, str, float, int, float, list[float]]] = []
        for u, v, d in g.edges(data=True):
            eid = int(d["edge_id"])
            rows.append(
                (
                    eid,
                    u,
                    v,
                    float(d["t"]),
                    int(d["is_laundering"]),
                    float(d.get("amount", 0.0)),
                    feats[eid],
                )
            )
        rows.sort(key=lambda r: r[3])  # by time -> leakage-safe ordering
        self.edge_ids = np.array([r[0] for r in rows], dtype=np.int64)
        self.src = [r[1] for r in rows]
        self.dst = [r[2] for r in rows]
        self.times = np.array([r[3] for r in rows], dtype=np.float64)
        self.y = np.array([r[4] for r in rows], dtype=np.int64)
        self.amounts = np.array([r[5] for r in rows], dtype=np.float64)
        self.X = np.array([r[6] for r in rows], dtype=np.float32)
        self.feature_names = list(FEATURE_NAMES)
        self.hits: dict[str, list[MotifHit]] = hits
        self.motif_edge_map: dict[str, set[int]] = {
            motif: {e for h in hs for e in h.edge_ids} for motif, hs in hits.items()
        }

    def __len__(self) -> int:
        """Number of edges."""
        return int(self.edge_ids.size)
