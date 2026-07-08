"""Unit tests for the motif detectors — explicit, small, deterministic graphs."""

from __future__ import annotations

from conftest import make_graph

from src import motifs
from src.config import MotifWindows


def test_fan_out_detects_one_to_many():
    g = make_graph([("A", "B", 0, 10), ("A", "C", 1, 10), ("A", "D", 2, 10)])
    hits = motifs.detect_fan_out(g, window_h=24, min_degree=3)
    assert any(h.center == "A" for h in hits)
    # a single edge is never a fan-out
    g2 = make_graph([("A", "B", 0, 10)])
    assert motifs.detect_fan_out(g2, window_h=24, min_degree=3) == []


def test_fan_out_respects_time_window():
    # three destinations, but spread far outside a 24h window -> no fan-out
    g = make_graph([("A", "B", 0, 10), ("A", "C", 100, 10), ("A", "D", 200, 10)])
    hits = motifs.detect_fan_out(g, window_h=24, min_degree=3)
    assert hits == []


def test_fan_in_detects_many_to_one():
    g = make_graph([("B", "A", 0, 10), ("C", "A", 1, 10), ("D", "A", 2, 10)])
    hits = motifs.detect_fan_in(g, window_h=24, min_degree=3)
    assert any(h.center == "A" for h in hits)


def test_cycle_requires_increasing_time():
    g = make_graph([("A", "B", 0, 5), ("B", "C", 1, 5), ("C", "A", 2, 5)])
    hits = motifs.detect_cycles(g, max_len=6, window_h=168)
    assert len(hits) >= 1
    # same topology but times decreasing around the loop -> not a temporal cycle
    bad = make_graph([("A", "B", 5, 5), ("B", "C", 1, 5), ("C", "A", 0, 5)])
    assert motifs.detect_cycles(bad, max_len=6, window_h=168) == []


def test_gather_scatter():
    edges = [
        ("S1", "H", 0, 10),
        ("S2", "H", 1, 10),
        ("S3", "H", 2, 10),
        ("H", "D1", 3, 9),
        ("H", "D2", 4, 9),
        ("H", "D3", 5, 9),
    ]
    hits = motifs.detect_gather_scatter(make_graph(edges), window_h=72, min_degree=3)
    assert any(h.center == "H" for h in hits)


def test_scatter_gather():
    edges = [
        ("S", "M1", 0, 10),
        ("S", "M2", 1, 10),
        ("S", "M3", 2, 10),
        ("M1", "C", 3, 9),
        ("M2", "C", 4, 9),
        ("M3", "C", 5, 9),
    ]
    hits = motifs.detect_scatter_gather(make_graph(edges), window_h=72, min_degree=3)
    assert any(h.center == "S" for h in hits)


def test_edge_features_shape_and_flags():
    edges = [("A", "B", 0, 10), ("A", "C", 1, 10), ("A", "D", 2, 10), ("B", "A", 3, 10)]
    g = make_graph(edges)
    feats, hits = motifs.edge_features(g, MotifWindows(), min_degree=3)
    assert set(feats.keys()) == {0, 1, 2, 3}
    for vec in feats.values():
        assert len(vec) == len(motifs.FEATURE_NAMES)
    # reciprocity: A->B (edge 0) has a reverse B->A -> reciprocity feature == 1
    recip_idx = motifs.FEATURE_NAMES.index("reciprocity")
    assert feats[0][recip_idx] == 1.0
    assert "fan_out" in hits
