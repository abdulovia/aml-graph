"""FastAPI service exposing the AML-Graph laundering-chain tracer.

Mirrors ``app/streamlit_app.py``: it consumes the scored-edge artifact the
pipeline writes to ``data/processed/edges_scored.parquet`` (one row per scored
test-set edge) and reuses :mod:`src.narrative` for the explanation.

Endpoints:

* ``GET /health`` — liveness plus whether the scored-edge artifact is present.
* ``POST /score_subgraph`` — given an account id (and optional hop count),
  return the ego-chain edges with per-edge risk scores and a deterministic
  template narrative.

The artifact is optional at import time: if it is missing, the scoring endpoint
returns HTTP 503 with a helpful message instead of crashing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.narrative import ChainFacts, template_narrative  # noqa: E402

SCORED = config.DATA_PROCESSED / "edges_scored.parquet"

app = FastAPI(
    title="AML-Graph API",
    version="0.1.0",
    description="Explainable money-laundering chain tracer over transaction graphs.",
)


class ScoreRequest(BaseModel):
    """Request body for :func:`score_subgraph`."""

    account: str = Field(..., description="Account id to trace (matches src/dst in the data).")
    hops: int = Field(1, ge=1, le=2, description="Neighbourhood radius around the account.")
    threshold: float = Field(
        0.5, ge=0.0, le=1.0, description="Risk threshold above which an edge joins the chain."
    )


class ScoredEdge(BaseModel):
    """A single scored transaction edge in the ego subgraph."""

    edge_id: int
    src: str
    dst: str
    amount: float
    score: float
    is_laundering: int
    in_chain: bool


class ScoreResponse(BaseModel):
    """Response body for :func:`score_subgraph`."""

    account: str
    hops: int
    threshold: float
    n_edges: int
    n_chain_edges: int
    edges: list[ScoredEdge]
    narrative: str


def _load_edges() -> pl.DataFrame | None:
    """Load the scored-edge artifact if it exists, else ``None``."""
    if not SCORED.exists():
        return None
    return pl.read_parquet(SCORED)


def _ego_edges(df: pl.DataFrame, account: str, hops: int) -> pl.DataFrame:
    """Return edges within ``hops`` of ``account`` (both endpoints in the frontier)."""
    keep = {account}
    for _ in range(hops):
        mask = df.filter(pl.col("src").is_in(keep) | pl.col("dst").is_in(keep))
        keep |= set(mask.get_column("src").to_list()) | set(mask.get_column("dst").to_list())
    return df.filter(pl.col("src").is_in(keep) & pl.col("dst").is_in(keep))


@app.get("/health")
def health() -> dict[str, object]:
    """Report service liveness and whether the scored-edge artifact is available."""
    return {"status": "ok", "artifact_present": SCORED.exists(), "artifact_path": str(SCORED)}


@app.post("/score_subgraph", response_model=ScoreResponse)
def score_subgraph(req: ScoreRequest) -> ScoreResponse:
    """Trace the ego subgraph around an account and explain its risk chain.

    Args:
        req: The account id, hop radius and risk threshold.

    Returns:
        The scored ego edges plus a template narrative for the flagged chain.

    Raises:
        HTTPException: 503 if the scored-edge artifact has not been produced yet,
            404 if the account is absent from the artifact.

    """
    df = _load_edges()
    if df is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Scored-edge artifact not found at {SCORED}. Run the pipeline first: "
                'python -c "from src import pipeline; pipeline.run_mvp(use_llm=False)"'
            ),
        )

    sub = _ego_edges(df, req.account, req.hops)
    if sub.is_empty():
        raise HTTPException(
            status_code=404, detail=f"Account {req.account!r} not found in artifact."
        )

    edges: list[ScoredEdge] = []
    chain_accounts: set[str] = set()
    chain_amounts: list[float] = []
    chain_scores: list[float] = []
    chain_times: list[float] = []
    for row in sub.iter_rows(named=True):
        score = float(row.get("score", 0.0))
        in_chain = score >= req.threshold
        edges.append(
            ScoredEdge(
                edge_id=int(row["edge_id"]),
                src=row["src"],
                dst=row["dst"],
                amount=float(row.get("amount", 0.0)),
                score=score,
                is_laundering=int(row.get("is_laundering", 0)),
                in_chain=in_chain,
            )
        )
        if in_chain:
            chain_accounts.update((row["src"], row["dst"]))
            chain_amounts.append(float(row.get("amount", 0.0)))
            chain_scores.append(score)
            chain_times.append(float(row.get("t", 0.0)))

    n_chain = len(chain_amounts)
    span_hours = (max(chain_times) - min(chain_times)) / 3600.0 if len(chain_times) > 1 else 0.0
    facts = ChainFacts(
        motif_type="suspicious_chain",
        n_accounts=len(chain_accounts),
        n_transactions=n_chain,
        total_amount=float(sum(chain_amounts)),
        span_hours=span_hours,
        confidence=sum(chain_scores) / n_chain if n_chain else 0.0,
        focus_account=req.account,
    )
    narrative = (
        template_narrative(facts)
        if n_chain
        else "No edges above the current risk threshold for this account."
    )

    return ScoreResponse(
        account=req.account,
        hops=req.hops,
        threshold=req.threshold,
        n_edges=len(edges),
        n_chain_edges=n_chain,
        edges=edges,
        narrative=narrative,
    )
