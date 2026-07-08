"""Shared test fixtures/helpers for building tiny temporal multigraphs."""

from __future__ import annotations

import networkx as nx

HOUR = 3600.0


def make_graph(edges: list[tuple[str, str, float, float]]) -> nx.MultiDiGraph:
    """Build a MultiDiGraph from ``(src, dst, t_hours, amount)`` tuples.

    Timestamps are given in hours for readability and stored as epoch seconds.
    """
    g = nx.MultiDiGraph()
    for i, (u, v, t_h, amt) in enumerate(edges):
        g.add_edge(u, v, edge_id=i, t=t_h * HOUR, amount=amt, is_laundering=0)
    return g
