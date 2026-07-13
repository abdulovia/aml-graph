"""Load the declarative ``configs/config.yaml`` into a :class:`~src.config.RunConfig`.

This is a thin convenience helper — the pipeline still defaults to
:data:`src.config.DEFAULT_RUN`. It lets an operator keep run parameters in YAML
(and override them from the CLI / CI) without refactoring the dataclass-based
config module. PyYAML is imported lazily so the light test environment (numpy +
networkx only) is unaffected.
"""

from __future__ import annotations

from pathlib import Path

from src.config import ROOT, MotifWindows, RunConfig

DEFAULT_CONFIG_PATH: Path = ROOT / "configs" / "config.yaml"


def load(path: Path | str | None = None) -> RunConfig:
    """Read a YAML config file into a :class:`RunConfig`.

    Missing keys fall back to the :class:`RunConfig` / :class:`MotifWindows`
    defaults, so a partial YAML is valid.

    Args:
        path: Path to the YAML file; defaults to ``configs/config.yaml``.

    Returns:
        A populated, frozen :class:`RunConfig`.

    """
    import yaml  # lazy: keep the light test env free of the PyYAML dependency

    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    raw = yaml.safe_load(cfg_path.read_text()) or {}

    run = raw.get("run", {})
    win = raw.get("windows", {})
    defaults = RunConfig()
    default_windows = MotifWindows()

    windows = MotifWindows(
        fan_window_h=win.get("fan_window_h", default_windows.fan_window_h),
        cycle_window_h=win.get("cycle_window_h", default_windows.cycle_window_h),
        gather_scatter_window_h=win.get(
            "gather_scatter_window_h", default_windows.gather_scatter_window_h
        ),
    )
    return RunConfig(
        sample_edges=run.get("sample_edges", defaults.sample_edges),
        temporal_train_frac=run.get("temporal_train_frac", defaults.temporal_train_frac),
        min_fan_degree=run.get("min_fan_degree", defaults.min_fan_degree),
        precision_at_k=tuple(run.get("precision_at_k", defaults.precision_at_k)),
        fixed_recall=run.get("fixed_recall", defaults.fixed_recall),
        windows=windows,
    )
