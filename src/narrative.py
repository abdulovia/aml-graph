"""Auto-narrative generation for a detected laundering chain.

Two modes:

* :func:`template_narrative` — deterministic, offline, reproducible. The MVP
  default; requires no API key and is ruff/CI-safe.
* :func:`llm_narrative` — a single, cached call to the Anthropic API that turns
  the same structured facts into a polished SAR-style narrative. Uses the cheap
  Haiku tier and caches the result to disk so it runs at most once (the project
  budget is small — do not burn it on repeats).

No secret is ever hard-coded: the API key is read from ``ANTHROPIC_API_KEY``.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from src import config

NARRATIVE_MODEL = "claude-haiku-4-5"  # cheapest tier ($1/$5 per 1M) — budget-safe
_CACHE = config.OUTPUTS / "narrative_cache.json"


@dataclass
class ChainFacts:
    """Structured facts extracted from a detected motif chain.

    Attributes:
        motif_type: The dominant motif (e.g. ``"fan_out"``, ``"gather_scatter"``).
        n_accounts: Distinct accounts involved.
        n_transactions: Transactions in the chain.
        total_amount: Total value moved (payment currency, summed).
        span_hours: Time span of the chain in hours.
        confidence: Model confidence for the flagged edges in ``[0, 1]``.
        focus_account: The account the analyst queried.

    """

    motif_type: str
    n_accounts: int
    n_transactions: int
    total_amount: float
    span_hours: float
    confidence: float
    focus_account: str


def template_narrative(facts: ChainFacts) -> str:
    """Render a deterministic SAR-style narrative from ``facts``."""
    pretty = facts.motif_type.replace("_", "-")
    rec = "FILE SAR" if facts.confidence >= 0.5 else "ENHANCED REVIEW"
    return (
        f"Account {facts.focus_account} sits at the centre of a {pretty} pattern: "
        f"{facts.n_transactions} transactions across {facts.n_accounts} accounts moved "
        f"~{facts.total_amount:,.0f} over {facts.span_hours:.1f} h. The structuring — "
        f"rapid dispersal/consolidation inconsistent with normal activity — matches a "
        f"known laundering typology. Model confidence {facts.confidence:.0%}. "
        f"Recommendation: {rec}."
    )


def _facts_key(facts: ChainFacts) -> str:
    """Stable cache key for a set of facts."""
    blob = json.dumps(asdict(facts), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _load_cache() -> dict[str, str]:
    if _CACHE.exists():
        return json.loads(_CACHE.read_text())
    return {}


def _save_cache(cache: dict[str, str]) -> None:
    _CACHE.write_text(json.dumps(cache, indent=2, ensure_ascii=False))


def llm_narrative(facts: ChainFacts, use_cache: bool = True) -> str:
    """Generate a narrative via one cached Anthropic API call.

    Falls back to :func:`template_narrative` when no API key is configured or
    the call fails, so the notebook always produces a narrative.

    Args:
        facts: The structured chain facts.
        use_cache: Reuse a previously generated narrative for identical facts.

    Returns:
        The generated (or cached, or fallback) narrative text.

    """
    key = _facts_key(facts)
    cache = _load_cache()
    if use_cache and key in cache:
        return cache[key]

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return template_narrative(facts)

    try:
        import anthropic

        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        prompt = (
            "You are an AML compliance analyst. Write a concise (<=90 words) "
            "suspicious-activity narrative for a regulator, in a factual tone, "
            "based strictly on these facts. End with a clear recommendation "
            "(SAR filing vs enhanced review).\n\n"
            f"Facts (JSON):\n{json.dumps(asdict(facts), ensure_ascii=False)}"
        )
        message = client.messages.create(
            model=NARRATIVE_MODEL,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        text = next((b.text for b in message.content if b.type == "text"), "").strip()
    except Exception:  # noqa: BLE001 - never let the demo crash on API issues
        return template_narrative(facts)

    if not text:
        return template_narrative(facts)
    cache[key] = text
    _save_cache(cache)
    return text


def facts_to_json(facts: ChainFacts, path: Path | None = None) -> None:
    """Persist the facts used for a narrative alongside the metrics."""
    path = path or (config.OUTPUTS / "narrative_facts.json")
    path.write_text(json.dumps(asdict(facts), indent=2, ensure_ascii=False))
