"""Figure generation for the MVP (six PNGs for the write-up and slides).

Each function is self-contained, uses a consistent style, and writes a single
PNG into ``outputs/figures``. Kept out of the notebook so the plots are
testable and re-runnable.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless, deterministic
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from src import config  # noqa: E402
from src.narrative import ChainFacts  # noqa: E402

# muted, colour-blind-safe palette
ILLICIT = "#c0392b"
LICIT = "#7f8c8d"
GNN_C = "#2c3e50"
LGBM_C = "#2980b9"
RAND_C = "#bdc3c7"
HYBRID_C = "#27ae60"

plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True, "grid.alpha": 0.25})


def _save(fig: plt.Figure, name: str) -> None:
    """Save a figure into the figures directory and close it."""
    config.ensure_dirs()
    fig.savefig(config.FIGURES / name, bbox_inches="tight")
    plt.close(fig)


def fig01_eda(df: pl.DataFrame) -> None:
    """EDA overview: imbalance, degree distributions, volume over time, formats."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    # (a) class imbalance
    frac = df.get_column("is_laundering").mean()
    axes[0, 0].bar(["licit", "illicit"], [1 - frac, frac], color=[LICIT, ILLICIT])
    axes[0, 0].set_title(f"Class balance (illicit = {frac:.3%})")
    axes[0, 0].set_ylabel("fraction")

    # (b) in/out-degree in log scale
    out_deg = df.group_by("src").len().get_column("len").to_numpy()
    in_deg = df.group_by("dst").len().get_column("len").to_numpy()
    bins = np.logspace(0, np.log10(max(out_deg.max(), in_deg.max()) + 1), 30)
    axes[0, 1].hist(out_deg, bins=bins, alpha=0.6, label="out-degree", color=LGBM_C)
    axes[0, 1].hist(in_deg, bins=bins, alpha=0.6, label="in-degree", color=ILLICIT)
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Account degree (log-log)")
    axes[0, 1].legend()

    # (c) transaction volume over time
    ts = df.select(pl.col("ts").dt.truncate("1d").alias("day")).group_by("day").len().sort("day")
    axes[1, 0].plot(ts.get_column("day").to_list(), ts.get_column("len").to_list(), color=GNN_C)
    axes[1, 0].set_title("Transactions per day")
    axes[1, 0].tick_params(axis="x", rotation=45)

    # (d) payment formats
    fmt = df.group_by("pay_format").len().sort("len", descending=True).head(8)
    axes[1, 1].barh(
        fmt.get_column("pay_format").to_list(), fmt.get_column("len").to_list(), color=LGBM_C
    )
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_title("Payment formats")

    fig.suptitle("IBM AMLworld HI-Small — EDA", fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig01_eda_imbalance.png")


def fig02_motif_freq(motif_illicit_rate: dict[str, tuple[int, int]]) -> None:
    """Illicit vs licit participation per motif.

    Args:
        motif_illicit_rate: motif_type -> (illicit_edges, licit_edges) among the
            edges the detector flagged for that motif.

    """
    motifs = list(motif_illicit_rate.keys())
    illicit = [motif_illicit_rate[m][0] for m in motifs]
    licit = [motif_illicit_rate[m][1] for m in motifs]
    x = np.arange(len(motifs))
    w = 0.4
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, licit, w, label="licit", color=LICIT)
    ax.bar(x + w / 2, illicit, w, label="illicit", color=ILLICIT)
    ax.set_yscale("symlog")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "-") for m in motifs], rotation=20)
    ax.set_ylabel("edges flagged (symlog)")
    ax.set_title("Motif participation: illicit vs licit edges")
    ax.legend()
    _save(fig, "fig02_motif_freq.png")


def fig03_pr_curve(y_true: np.ndarray, scores: dict[str, np.ndarray]) -> None:
    """Precision-recall curves for GNN vs LightGBM vs random."""
    colours = {"GNN": GNN_C, "LightGBM": LGBM_C, "Hybrid": HYBRID_C, "Random": RAND_C}
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, s in scores.items():
        prec, rec, _ = precision_recall_curve(y_true, s)
        ax.plot(rec, prec, label=name, color=colours.get(name), linewidth=2)
    baseline = float(y_true.mean())
    ax.axhline(baseline, ls="--", color="k", alpha=0.5, label=f"prevalence={baseline:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall (illicit class)")
    ax.legend()
    _save(fig, "fig03_pr_curve.png")


def fig04_precision_at_k(pk: dict[str, dict[str, float]]) -> None:
    """Precision@k bars for each model.

    Args:
        pk: model -> {"p@50": .., "p@100": ..}.

    """
    ks = list(next(iter(pk.values())).keys())
    models = list(pk.keys())
    x = np.arange(len(ks))
    w = 0.8 / max(len(models), 1)
    colours = {"GNN": GNN_C, "LightGBM": LGBM_C, "Hybrid": HYBRID_C, "Random": RAND_C}
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, m in enumerate(models):
        ax.bar(x + i * w, [pk[m][k] for k in ks], w, label=m, color=colours.get(m))
    ax.set_xticks(x + w * (len(models) - 1) / 2)
    ax.set_xticklabels(ks)
    ax.set_ylabel("precision")
    ax.set_title("Precision@k (analyst queue)")
    ax.legend()
    _save(fig, "fig04_precision_at_k.png")


def fig05_ring_trace(g: nx.MultiDiGraph, chain_edges: set[int], title: str) -> None:
    """Highlight a laundering chain (red) against surrounding grey transactions.

    Args:
        g: The full (or a local ego) multigraph.
        chain_edges: Edge ids belonging to the illicit chain.
        title: Figure title.

    """
    # keep the drawing readable: nodes touched by chain edges + their neighbours
    chain_nodes: set[str] = set()
    for u, v, d in g.edges(data=True):
        if int(d["edge_id"]) in chain_edges:
            chain_nodes.update((u, v))
    sub_nodes = set(chain_nodes)
    for n in chain_nodes:
        sub_nodes.update(list(g.successors(n))[:3])
        sub_nodes.update(list(g.predecessors(n))[:3])
    h = g.subgraph(sub_nodes)

    pos = nx.spring_layout(nx.DiGraph(h), seed=config.SEED, k=0.9)
    fig, ax = plt.subplots(figsize=(9, 7))
    nx.draw_networkx_nodes(h, pos, node_color=LICIT, node_size=140, alpha=0.5, ax=ax)
    nx.draw_networkx_nodes(
        h, pos, nodelist=list(chain_nodes), node_color=ILLICIT, node_size=320, ax=ax
    )
    grey_edges = [(u, v) for u, v, d in h.edges(data=True) if int(d["edge_id"]) not in chain_edges]
    red_edges = [(u, v) for u, v, d in h.edges(data=True) if int(d["edge_id"]) in chain_edges]
    nx.draw_networkx_edges(h, pos, edgelist=grey_edges, edge_color=LICIT, alpha=0.3, ax=ax)
    nx.draw_networkx_edges(
        h,
        pos,
        edgelist=red_edges,
        edge_color=ILLICIT,
        width=2.5,
        ax=ax,
        connectionstyle="arc3,rad=0.1",
    )
    nx.draw_networkx_labels(h, pos, labels={n: n for n in chain_nodes}, font_size=7, ax=ax)
    ax.set_title(title, fontweight="bold")
    ax.axis("off")
    _save(fig, "fig05_ring_trace.png")


def fig06_narrative_card(facts: ChainFacts, narrative: str) -> None:
    """Render a compliance 'alert card' summarising the chain + narrative."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axis("off")
    rec = "FILE SAR" if facts.confidence >= 0.5 else "ENHANCED REVIEW"
    band = ILLICIT if facts.confidence >= 0.5 else "#e67e22"

    ax.add_patch(plt.Rectangle((0, 0.88), 1, 0.12, transform=ax.transAxes, color=band))
    ax.text(
        0.02,
        0.94,
        f"⚠  AML ALERT — {facts.motif_type.replace('_', '-').upper()}",
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        color="white",
        va="center",
    )
    ax.text(
        0.98,
        0.94,
        rec,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        color="white",
        va="center",
        ha="right",
    )

    rows = [
        ("Focus account", facts.focus_account),
        ("Accounts involved", str(facts.n_accounts)),
        ("Transactions", str(facts.n_transactions)),
        ("Total moved", f"{facts.total_amount:,.0f}"),
        ("Time span", f"{facts.span_hours:.1f} h"),
        ("Model confidence", f"{facts.confidence:.0%}"),
    ]
    y = 0.80
    for label, val in rows:
        ax.text(0.04, y, label, transform=ax.transAxes, fontsize=10, color=LICIT)
        ax.text(0.42, y, val, transform=ax.transAxes, fontsize=10, fontweight="bold")
        y -= 0.075

    ax.text(0.04, 0.30, "Narrative", transform=ax.transAxes, fontsize=11, fontweight="bold")
    ax.text(
        0.04,
        0.02,
        narrative,
        transform=ax.transAxes,
        fontsize=9.5,
        va="bottom",
        wrap=True,
        bbox={"boxstyle": "round", "fc": "#f7f7f7", "ec": band},
    )
    _save(fig, "fig06_narrative_card.png")
