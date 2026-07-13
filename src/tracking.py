"""Optional MLflow experiment tracking.

Fully guarded: if ``mlflow`` is not installed (the light CI/test env) or no
tracking URI is configured, :func:`log_run` is a silent no-op. This keeps the
default pipeline path unchanged while allowing richer experiment tracking when
an operator opts in (``track=True`` and ``MLFLOW_TRACKING_URI`` set).

mlflow is imported lazily inside the function so importing this module never
pulls the heavy dependency.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Scalar metric keys worth logging from each model's summary dict.
_SCALAR_KEYS = (
    "minority_f1",
    "pr_auc",
    "review_fraction",
    "alert_reduction",
    "threshold",
    "p@50",
    "p@100",
    "p@500",
    "p@1000",
)


def _enabled() -> bool:
    """Return True only if mlflow is importable and a tracking URI is set."""
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        return False
    try:
        import mlflow  # noqa: F401
    except ImportError:
        return False
    return True


def log_run(
    metrics_dict: dict[str, Any],
    figures_dir: Path | str,
    experiment: str = "aml-graph",
) -> bool:
    """Log per-model metrics and figures to MLflow, if enabled.

    Args:
        metrics_dict: The ``result`` dict from :func:`src.pipeline.run_mvp`
            (per-model summaries plus a ``config`` block).
        figures_dir: Directory of PNG figures to attach as artifacts.
        experiment: MLflow experiment name.

    Returns:
        True if a run was logged, False if tracking was disabled / unavailable.

    """
    if not _enabled():
        return False

    import mlflow  # lazy import — heavy dependency

    figures_dir = Path(figures_dir)
    mlflow.set_experiment(experiment)
    with mlflow.start_run():
        cfg = metrics_dict.get("config", {})
        if isinstance(cfg, dict):
            mlflow.log_params({k: v for k, v in cfg.items() if not isinstance(v, dict)})

        for name, summary in metrics_dict.items():
            if not isinstance(summary, dict) or "minority_f1" not in summary:
                continue
            for key in _SCALAR_KEYS:
                value = summary.get(key)
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"{name}.{key}", float(value))

        if figures_dir.is_dir():
            for png in sorted(figures_dir.glob("*.png")):
                mlflow.log_artifact(str(png), artifact_path="figures")
    return True
