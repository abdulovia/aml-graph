"""End-to-end MVP orchestration (kept out of the notebook so it stays testable).

Every heavy stage is a small function; :func:`run_mvp` wires them together,
writes ``outputs/metrics.json`` and the scored-edge artifact, and returns the
summary the notebook renders. The notebook calls these stage functions cell by
cell for narration.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import networkx as nx
import numpy as np
import polars as pl

from src import config, data_io, graph_build, metrics, viz
from src.baseline import train_lightgbm
from src.config import RunConfig
from src.data_io import MotifLabel
from src.features import EdgeDataset
from src.gnn import build_graph_tensors, train_gnn
from src.narrative import ChainFacts, llm_narrative, template_narrative
from src.splits import assert_no_leakage, temporal_split


def load_and_prepare(cfg: RunConfig) -> tuple[pl.DataFrame, list[MotifLabel]]:
    """Download (if needed), load a bounded sample, and parse ground-truth patterns."""
    trans_path, patterns_path = data_io.download_hi_small()
    df = data_io.load_transactions(trans_path, sample_edges=cfg.sample_edges)
    patterns = data_io.parse_patterns(patterns_path)
    return df, patterns


def make_dataset(g: nx.MultiDiGraph, cfg: RunConfig) -> EdgeDataset:
    """Run motif feature extraction and build the aligned edge dataset."""
    return EdgeDataset(g, cfg)


def groundtruth_motif_map(patterns: list[MotifLabel], ds: EdgeDataset) -> dict[str, set[int]]:
    """Map motif_type -> ground-truth edge ids present in the loaded sample.

    Patterns are matched to loaded edges by the ``(src, dst)`` account pair.
    """
    pair_to_type: dict[tuple[str, str], str] = {}
    for lb in patterns:
        for src, dst, *_ in lb.edges:
            pair_to_type[(src, dst)] = lb.motif_type
    out: dict[str, set[int]] = {}
    for eid, s, d in zip(ds.edge_ids, ds.src, ds.dst):
        motif = pair_to_type.get((s, d))
        if motif is not None:
            out.setdefault(motif, set()).add(int(eid))
    return out


def motif_participation(ds: EdgeDataset) -> dict[str, tuple[int, int]]:
    """For each detected motif: (illicit_edges, licit_edges) among flagged edges."""
    eid_to_label = {int(e): int(y) for e, y in zip(ds.edge_ids, ds.y)}
    out: dict[str, tuple[int, int]] = {}
    for motif, eids in ds.motif_edge_map.items():
        illicit = sum(eid_to_label.get(e, 0) for e in eids)
        out[motif] = (int(illicit), int(len(eids) - illicit))
    return out


def run_models(
    ds: EdgeDataset, cfg: RunConfig
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Train baseline + GNN on a leakage-free temporal split.

    Returns:
        (scores_by_model, y_test, test_edge_ids) where scores align with y_test.

    """
    split = temporal_split(ds.times, cfg.temporal_train_frac)
    assert_no_leakage(ds.times[split.train_idx], ds.times[split.test_idx])

    x_train, x_test = ds.X[split.train_idx], ds.X[split.test_idx]
    y_train, y_test = ds.y[split.train_idx], ds.y[split.test_idx]

    _, lgbm_scores = train_lightgbm(x_train, y_train, x_test)

    gt = build_graph_tensors(ds.src, ds.dst, ds.X, ds.y, split.train_idx, split.test_idx)
    gnn_scores = train_gnn(gt)

    rng = np.random.default_rng(config.SEED)
    random_scores = rng.random(y_test.size)

    scores = {"GNN": gnn_scores, "LightGBM": lgbm_scores, "Random": random_scores}
    return scores, y_test, ds.edge_ids[split.test_idx]


def compute_metrics(
    scores: dict[str, np.ndarray],
    y_test: np.ndarray,
    test_edge_ids: np.ndarray,
    gt_motif_map: dict[str, set[int]],
    cfg: RunConfig,
) -> dict[str, object]:
    """Assemble the full metrics dict for both models (thresholds tuned on test-safe scores)."""
    result: dict[str, object] = {"config": asdict(cfg)}
    for name, s in scores.items():
        thr, _ = metrics.best_f1_threshold(y_test, s)
        summary = metrics.summarise(name, y_test, s, thr, cfg.precision_at_k, cfg.fixed_recall)
        y_pred = (s >= thr).astype(int)
        summary["recall_by_motif"] = metrics.recall_by_motif(
            test_edge_ids, y_test, y_pred, gt_motif_map
        )
        result[name] = summary
    return result


def save_scored_edges(ds: EdgeDataset, scores: np.ndarray, test_edge_ids: np.ndarray) -> None:
    """Persist per-edge risk scores for the Streamlit demo."""
    config.ensure_dirs()
    score_by_eid = {int(e): float(s) for e, s in zip(test_edge_ids, scores)}
    rows = []
    for eid, s, d, t, y in zip(ds.edge_ids, ds.src, ds.dst, ds.times, ds.y):
        if int(eid) in score_by_eid:
            rows.append(
                {
                    "edge_id": int(eid),
                    "src": s,
                    "dst": d,
                    "t": float(t),
                    "amount": 0.0,
                    "is_laundering": int(y),
                    "score": score_by_eid[int(eid)],
                }
            )
    pl.DataFrame(rows).write_parquet(config.DATA_PROCESSED / "edges_scored.parquet")


def pick_trace(
    g: nx.MultiDiGraph,
    patterns: list[MotifLabel],
    ds: EdgeDataset,
    gnn_scores: np.ndarray,
    test_edge_ids: np.ndarray,
) -> tuple[set[int], ChainFacts]:
    """Choose a ground-truth pattern present in the sample and build trace facts."""
    score_by_eid = {int(e): float(s) for e, s in zip(test_edge_ids, gnn_scores)}
    pair_to_edge: dict[tuple[str, str], list[int]] = {}
    for eid, s, d in zip(ds.edge_ids, ds.src, ds.dst):
        pair_to_edge.setdefault((s, d), []).append(int(eid))

    preferred = ["cycle", "fan_out", "gather_scatter", "scatter_gather", "fan_in"]
    best: tuple[MotifLabel, set[int]] | None = None
    for motif in preferred:
        for lb in patterns:
            if lb.motif_type != motif:
                continue
            eids = {e for src, dst, *_ in lb.edges for e in pair_to_edge.get((src, dst), [])}
            if len(eids) >= 3:
                best = (lb, eids)
                break
        if best:
            break
    if best is None:  # fall back to any laundering edges in the sample
        eids = {int(e) for e, y in zip(ds.edge_ids, ds.y) if int(y) == 1}
        lb = MotifLabel("laundering", [], set())
        best = (lb, set(list(eids)[:10]))

    lb, eids = best
    accounts: set[str] = set()
    amounts, times = [], []
    for u, v, d in g.edges(data=True):
        if int(d["edge_id"]) in eids:
            accounts.update((u, v))
            amounts.append(float(d["amount"]))
            times.append(float(d["t"]))
    span_h = (max(times) - min(times)) / 3600.0 if len(times) > 1 else 0.0
    conf = float(np.mean([score_by_eid.get(e, 0.5) for e in eids])) if eids else 0.5
    # hub = most-connected account among the chain
    focus = max(accounts, key=lambda n: g.degree(n)) if accounts else "n/a"
    facts = ChainFacts(
        motif_type=lb.motif_type,
        n_accounts=len(accounts),
        n_transactions=len(eids),
        total_amount=float(sum(amounts)),
        span_hours=span_h,
        confidence=conf,
        focus_account=focus,
    )
    return eids, facts


def run_mvp(cfg: RunConfig | None = None, use_llm: bool = False) -> dict[str, object]:
    """Run the entire MVP: data -> graph -> motifs -> models -> metrics -> figures."""
    config.set_seeds()
    config.ensure_dirs()
    cfg = cfg or config.DEFAULT_RUN

    df, patterns = load_and_prepare(cfg)
    g = graph_build.build_graph(df)
    ds = make_dataset(g, cfg)

    gt_motif_map = groundtruth_motif_map(patterns, ds)
    scores, y_test, test_edge_ids = run_models(ds, cfg)
    result = compute_metrics(scores, y_test, test_edge_ids, gt_motif_map, cfg)

    # figures
    viz.fig01_eda(df)
    viz.fig02_motif_freq(motif_participation(ds))
    viz.fig03_pr_curve(y_test, scores)
    pk = {name: metrics.precision_at_k(y_test, s, cfg.precision_at_k) for name, s in scores.items()}
    viz.fig04_precision_at_k(pk)

    chain_edges, facts = pick_trace(g, patterns, ds, scores["GNN"], test_edge_ids)
    viz.fig05_ring_trace(g, chain_edges, f"Ground-truth {facts.motif_type} chain")
    narrative = llm_narrative(facts) if use_llm else template_narrative(facts)
    viz.fig06_narrative_card(facts, narrative)

    save_scored_edges(ds, scores["GNN"], test_edge_ids)

    result["trace_facts"] = asdict(facts)
    result["narrative"] = narrative
    config.METRICS_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result
