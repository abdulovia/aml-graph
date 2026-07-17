"""Streamlit demo: enter an account, see its highlighted laundering chain + narrative.

Run with:  streamlit run app/streamlit_app.py

Consumes the artifact the notebook writes to
``data/processed/edges_scored.parquet`` (one row per scored test-set edge). If it
is missing, the app explains how to produce it rather than crashing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import polars as pl
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.narrative import ChainFacts, template_narrative  # noqa: E402

SCORED = config.DATA_PROCESSED / "edges_scored.parquet"
ILLICIT, LICIT = "#c0392b", "#7f8c8d"


@st.cache_data
def load_edges() -> pl.DataFrame | None:
    """Load the scored-edge artifact if present."""
    if not SCORED.exists():
        return None
    return pl.read_parquet(SCORED)


def ego_graph(df: pl.DataFrame, account: str, hops: int = 1) -> nx.MultiDiGraph:
    """Build a small ego multigraph around ``account`` from the scored edges."""
    keep = {account}
    for _ in range(hops):
        mask = df.filter(pl.col("src").is_in(keep) | pl.col("dst").is_in(keep))
        keep |= set(mask.get_column("src").to_list()) | set(mask.get_column("dst").to_list())
    sub = df.filter(pl.col("src").is_in(keep) & pl.col("dst").is_in(keep))
    g = nx.MultiDiGraph()
    for row in sub.iter_rows(named=True):
        g.add_edge(
            row["src"],
            row["dst"],
            edge_id=row["edge_id"],
            amount=row["amount"],
            t=row["t"],
            score=row.get("score", 0.0),
            is_laundering=row.get("is_laundering", 0),
        )
    return g


def draw(g: nx.MultiDiGraph, focus: str, threshold: float) -> plt.Figure:
    """Render the ego graph with high-score (chain) edges in red."""
    pos = nx.spring_layout(nx.DiGraph(g), seed=config.SEED, k=0.9)
    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw_networkx_nodes(g, pos, node_color=LICIT, node_size=120, alpha=0.5, ax=ax)
    nx.draw_networkx_nodes(g, pos, nodelist=[focus], node_color=ILLICIT, node_size=360, ax=ax)
    hot = [(u, v) for u, v, d in g.edges(data=True) if d.get("score", 0) >= threshold]
    cold = [(u, v) for u, v, d in g.edges(data=True) if d.get("score", 0) < threshold]
    nx.draw_networkx_edges(g, pos, edgelist=cold, edge_color=LICIT, alpha=0.3, ax=ax)
    nx.draw_networkx_edges(g, pos, edgelist=hot, edge_color=ILLICIT, width=2.5, ax=ax)
    ax.set_title(f"Transaction chain around {focus}")
    ax.axis("off")
    return fig


def main() -> None:
    """Streamlit entry point."""
    st.set_page_config(page_title="AML-Graph demo", layout="wide")
    st.title("AML-Graph — explainable laundering-chain tracer")

    df = load_edges()
    if df is None:
        st.warning(
            "No scored edges found. Run `notebooks/mvp_aml.ipynb` first — it writes "
            f"`{SCORED}` with per-edge risk scores."
        )
        return

    # Rank accounts by their riskiest incident edge so the demo opens on a
    # genuinely suspicious account instead of an arbitrary alphabetical one.
    risk = (
        pl.concat(
            [
                df.select(pl.col("src").alias("account"), "score"),
                df.select(pl.col("dst").alias("account"), "score"),
            ]
        )
        .group_by("account")
        .agg(pl.col("score").max().alias("max_score"))
        .sort("max_score", descending=True)
    )
    accounts = risk.get_column("account").to_list()

    col1, col2 = st.columns([1, 3])
    with col1:
        account = st.selectbox("Account (riskiest first)", accounts)
        threshold = st.slider("Risk threshold", 0.0, 1.0, 0.5, 0.05)
        hops = st.slider("Neighbourhood hops", 1, 2, 1)

    g = ego_graph(df, account, hops)
    with col2:
        st.pyplot(draw(g, account, threshold))

    chain_edges = [d for _, _, d in g.edges(data=True) if d.get("score", 0) >= threshold]
    if chain_edges:
        amounts = [d["amount"] for d in chain_edges]
        scores = [d["score"] for d in chain_edges]
        times = [d["t"] for d in chain_edges]
        chain_accounts = {
            n for u, v, d in g.edges(data=True) if d.get("score", 0) >= threshold for n in (u, v)
        }
        span_h = (max(times) - min(times)) / 3600.0 if len(times) > 1 else 0.0
        facts = ChainFacts(
            motif_type="suspicious_chain",
            n_accounts=len(chain_accounts),
            n_transactions=len(chain_edges),
            total_amount=float(sum(amounts)),
            span_hours=span_h,
            confidence=float(sum(scores) / len(scores)),
            focus_account=account,
        )
        st.subheader("Auto-narrative")
        st.info(template_narrative(facts))
    else:
        st.success("No high-risk chain above the current threshold for this account.")


if __name__ == "__main__":
    main()
