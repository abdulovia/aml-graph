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

    accounts = sorted(set(df.get_column("src").to_list()) | set(df.get_column("dst").to_list()))
    col1, col2 = st.columns([1, 3])
    with col1:
        account = st.selectbox("Account", accounts)
        threshold = st.slider("Risk threshold", 0.0, 1.0, 0.5, 0.05)
        hops = st.slider("Neighbourhood hops", 1, 2, 1)

    g = ego_graph(df, account, hops)
    with col2:
        st.pyplot(draw(g, account, threshold))

    chain = {int(d["edge_id"]) for _, _, d in g.edges(data=True) if d.get("score", 0) >= threshold}
    if chain:
        amounts = [d["amount"] for _, _, d in g.edges(data=True) if int(d["edge_id"]) in chain]
        facts = ChainFacts(
            motif_type="fan_out",
            n_accounts=g.number_of_nodes(),
            n_transactions=len(chain),
            total_amount=float(sum(amounts)),
            span_hours=24.0,
            confidence=threshold,
            focus_account=account,
        )
        st.subheader("Auto-narrative")
        st.info(template_narrative(facts))
    else:
        st.success("No high-risk chain above the current threshold for this account.")


if __name__ == "__main__":
    main()
