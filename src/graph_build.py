"""Build the directed temporal multigraph from the transaction table."""

from __future__ import annotations

import networkx as nx
import polars as pl


def build_graph(df: pl.DataFrame) -> nx.MultiDiGraph:
    """Construct a :class:`networkx.MultiDiGraph` from a transaction DataFrame.

    Node = account (``Bank_Account``). Edge = one transaction, carrying time as
    float epoch seconds (``t``), ``amount``, currency/format and the
    ``is_laundering`` label plus a stable ``edge_id``.

    Args:
        df: Output of :func:`src.data_io.load_transactions` (must contain
            ``src``, ``dst``, ``ts``, ``amount_paid``, ``is_laundering``,
            ``edge_id``).

    Returns:
        The populated multigraph.

    """
    g = nx.MultiDiGraph()
    epoch = df.select(pl.col("ts").dt.epoch(time_unit="s").alias("t")).to_series().to_list()
    src = df.get_column("src").to_list()
    dst = df.get_column("dst").to_list()
    amount = df.get_column("amount_paid").cast(pl.Float64).to_list()
    fmt = df.get_column("pay_format").to_list()
    cur = df.get_column("pay_currency").to_list()
    lbl = df.get_column("is_laundering").to_list()
    eid = df.get_column("edge_id").to_list()

    for i in range(df.height):
        g.add_edge(
            src[i],
            dst[i],
            edge_id=int(eid[i]),
            t=float(epoch[i]) if epoch[i] is not None else 0.0,
            amount=float(amount[i]) if amount[i] is not None else 0.0,
            fmt=fmt[i],
            currency=cur[i],
            is_laundering=int(lbl[i]) if lbl[i] is not None else 0,
        )
    return g


def edge_frame(g: nx.MultiDiGraph) -> pl.DataFrame:
    """Flatten graph edges back into a Polars DataFrame keyed by ``edge_id``."""
    rows = [
        {
            "edge_id": int(d["edge_id"]),
            "src": u,
            "dst": v,
            "t": float(d["t"]),
            "amount": float(d["amount"]),
            "is_laundering": int(d["is_laundering"]),
        }
        for u, v, d in g.edges(data=True)
    ]
    return pl.DataFrame(rows).sort("t")
