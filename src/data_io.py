"""Data acquisition and loading for IBM AMLworld (HI-Small).

Handles the Kaggle download, memory-friendly CSV loading via Polars, and
parsing of the ground-truth ``*_Patterns.txt`` file into typed motif labels.
"""

from __future__ import annotations

import os
import re
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from src import config

# Canonical column names (the raw CSV ships two columns literally named
# "Account", which breaks name-based access — we rename positionally).
COLUMNS: list[str] = [
    "timestamp",
    "from_bank",
    "from_account",
    "to_bank",
    "to_account",
    "amount_received",
    "recv_currency",
    "amount_paid",
    "pay_currency",
    "pay_format",
    "is_laundering",
]

_TS_FORMAT = "%Y/%m/%d %H:%M"
_BEGIN_RE = re.compile(r"BEGIN LAUNDERING ATTEMPT\s*-\s*(.+)", re.IGNORECASE)
_END_RE = re.compile(r"END LAUNDERING ATTEMPT", re.IGNORECASE)


def node_id(bank: str, account: str) -> str:
    """Build the canonical ``Bank_Account`` node identifier."""
    return f"{bank}_{account}"


# --- Kaggle download -------------------------------------------------------
def _configure_kaggle_env() -> None:
    """Surface a Kaggle token from the standard locations into the env.

    Supports both the classic ``~/.kaggle/kaggle.json`` and the newer
    ``KAGGLE_API_TOKEN`` / ``~/.kaggle/access_token`` mechanisms. No secret is
    ever written to the repository.
    """
    if os.environ.get("KAGGLE_API_TOKEN"):
        return
    token_file = Path.home() / ".kaggle" / "access_token"
    if token_file.exists():
        os.environ["KAGGLE_API_TOKEN"] = token_file.read_text().strip()


def download_hi_small(force: bool = False) -> tuple[Path, Path]:
    """Download HI-Small transactions + patterns from Kaggle if missing.

    Args:
        force: Re-download even if the files already exist locally.

    Returns:
        Paths to ``HI-Small_Trans.csv`` and ``HI-Small_Patterns.txt``.

    """
    config.ensure_dirs()
    _configure_kaggle_env()
    trans = config.DATA_RAW / config.HI_SMALL_TRANS
    patterns = config.DATA_RAW / config.HI_SMALL_PATTERNS
    if trans.exists() and patterns.exists() and not force:
        return trans, patterns

    for fname in (config.HI_SMALL_TRANS, config.HI_SMALL_PATTERNS):
        subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                config.KAGGLE_DATASET,
                "-f",
                fname,
                "-p",
                str(config.DATA_RAW),
            ],
            check=True,
        )
        zpath = config.DATA_RAW / f"{fname}.zip"
        if zpath.exists():
            with zipfile.ZipFile(zpath) as zf:
                zf.extractall(config.DATA_RAW)
            zpath.unlink()
    return trans, patterns


# --- Transaction loading ---------------------------------------------------
def load_transactions(
    path: Path | None = None,
    sample_edges: int | None = None,
) -> pl.DataFrame:
    """Load the HI-Small transaction CSV into a typed Polars DataFrame.

    A head-sample (first ``sample_edges`` rows) keeps the run memory-bounded.
    Because the CSV is time-ordered, a head-sample is also a contiguous early
    time slice, which is convenient but *not* used as the split itself (the
    temporal split is computed explicitly downstream).

    Args:
        path: CSV path; defaults to the standard raw location.
        sample_edges: Optional cap on the number of rows read.

    Returns:
        A DataFrame with canonical columns, a parsed ``ts`` datetime column and
        derived ``src``/``dst`` node ids.

    """
    path = path or (config.DATA_RAW / config.HI_SMALL_TRANS)
    # Read every column as Utf8: bank codes carry leading zeros ("010") that an
    # int parse would drop, breaking the match against the string-typed Patterns
    # file. Numeric columns are cast explicitly in _finalise / downstream.
    df = pl.read_csv(path, has_header=True, new_columns=COLUMNS, infer_schema_length=0)
    if sample_edges and 0 < sample_edges < df.height:
        # Keep EVERY illicit transaction (they carry the laundering motifs, and
        # a head-sample would drop most positives), and randomly subsample licit
        # edges to the target size. This inflates prevalence versus the full set;
        # the true and sampled rates are both recorded in metrics.json. The
        # temporal split is still valid — it splits the result by timestamp.
        illicit = df.filter(pl.col("is_laundering") == "1")
        licit = df.filter(pl.col("is_laundering") == "0")
        n_licit = max(sample_edges - illicit.height, 0)
        licit_s = licit.sample(n=min(n_licit, licit.height), seed=config.SEED)
        df = pl.concat([illicit, licit_s])
    return _finalise(df)


def _finalise(df: pl.DataFrame) -> pl.DataFrame:
    """Attach parsed timestamp, node ids and a stable edge index."""
    df = df.with_columns(
        pl.col("timestamp").str.to_datetime(_TS_FORMAT, strict=False).alias("ts"),
        (pl.col("from_bank").cast(pl.Utf8) + "_" + pl.col("from_account").cast(pl.Utf8)).alias(
            "src"
        ),
        (pl.col("to_bank").cast(pl.Utf8) + "_" + pl.col("to_account").cast(pl.Utf8)).alias("dst"),
        pl.col("is_laundering").cast(pl.Int8),
    )
    df = df.sort("ts").with_row_index("edge_id")
    return df


# --- Patterns ground-truth -------------------------------------------------
@dataclass
class MotifLabel:
    """A single ground-truth laundering pattern from the Patterns file.

    Attributes:
        motif_type: Normalised type, e.g. ``"fan_out"``, ``"cycle"``.
        edges: Ordered list of ``(src, dst, ts_str, amount, is_laundering)``.
        accounts: Set of node ids participating in the pattern.

    """

    motif_type: str
    edges: list[tuple[str, str, str, float, int]]
    accounts: set[str]


def _normalise_type(raw: str) -> str:
    """Map a raw Patterns header label to a base snake_case motif type.

    Headers look like ``FAN-OUT:  Max 16-degree Fan-Out`` — we keep only the
    base type before the colon (``fan_out``).
    """
    base = raw.split(":")[0]
    return base.strip().lower().replace(" ", "_").replace("-", "_")


def parse_patterns(path: Path | None = None) -> list[MotifLabel]:
    """Parse ``HI-Small_Patterns.txt`` into a list of :class:`MotifLabel`.

    The file groups labelled laundering transactions in blocks delimited by
    ``BEGIN LAUNDERING ATTEMPT - <TYPE>`` / ``END LAUNDERING ATTEMPT`` lines.

    Args:
        path: Patterns file path; defaults to the standard raw location.

    Returns:
        One :class:`MotifLabel` per pattern block.

    """
    path = path or (config.DATA_RAW / config.HI_SMALL_PATTERNS)
    labels: list[MotifLabel] = []
    current_type: str | None = None
    edges: list[tuple[str, str, str, float, int]] = []
    accounts: set[str] = set()

    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            begin = _BEGIN_RE.search(line)
            if begin:
                current_type = _normalise_type(begin.group(1))
                edges, accounts = [], set()
                continue
            if _END_RE.search(line):
                if current_type is not None and edges:
                    labels.append(MotifLabel(current_type, edges, set(accounts)))
                current_type = None
                continue
            if current_type is None:
                continue
            parsed = _parse_pattern_row(line)
            if parsed is not None:
                src, dst, ts_str, amount, is_ml = parsed
                edges.append((src, dst, ts_str, amount, is_ml))
                accounts.update((src, dst))
    return labels


def _parse_pattern_row(line: str) -> tuple[str, str, str, float, int] | None:
    """Parse one CSV transaction row from within a pattern block."""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 11 or parts[0].lower().startswith("timestamp"):
        return None
    try:
        ts_str = parts[0]
        src = node_id(parts[1], parts[2])
        dst = node_id(parts[3], parts[4])
        amount = float(parts[7]) if parts[7] else 0.0
        is_ml = int(float(parts[10])) if parts[10] else 1
    except (ValueError, IndexError):
        return None
    return src, dst, ts_str, amount, is_ml


def patterns_summary(labels: list[MotifLabel]) -> pl.DataFrame:
    """Count ground-truth patterns and edges per motif type."""
    rows = [
        {"motif_type": lb.motif_type, "n_edges": len(lb.edges), "n_accounts": len(lb.accounts)}
        for lb in labels
    ]
    if not rows:
        return pl.DataFrame({"motif_type": [], "n_patterns": [], "n_edges": []})
    return (
        pl.DataFrame(rows)
        .group_by("motif_type")
        .agg(
            pl.len().alias("n_patterns"),
            pl.col("n_edges").sum().alias("n_edges"),
        )
        .sort("n_patterns", descending=True)
    )
