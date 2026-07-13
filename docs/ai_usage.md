# AI usage

This document discloses, for the contest's *AI application* criterion, how AI
agents were used to build AML-Graph — honestly, with the boundary between
AI-assisted and human-directed work made explicit.

## Summary

AML-Graph was built with **Claude Code** (Anthropic's agentic CLI) acting as a
pair engineer under human direction. The human set the problem, the modelling
approach, the honesty constraints (prevalence disclosure, no committed secrets)
and reviewed every artifact; the agent drafted code, tests, infrastructure and
documentation, and ran the tooling to verify them.

## What was AI-assisted

| Area | AI-assisted | Human-directed |
|------|-------------|----------------|
| Core `src/` modules (data_io, graph_build, splits, motifs, features, metrics, baseline, gnn, viz, pipeline) | Drafting, refactoring, docstrings | Algorithm/metric choice, motif definitions, leakage-safety requirement |
| Tests (`tests/test_motifs.py`, `tests/test_no_leakage.py`) | Test authoring | Which invariants matter (no temporal leakage, motif correctness) |
| LLM-narrative feature (`src/narrative.py`) | Template + cached Anthropic Haiku integration | Budget cap, cache-once policy, fallback-to-template rule |
| Engineering wrapper (Dockerfile, docker-compose, FastAPI, Makefile, DVC, MLflow hook, pre-commit, CI, README) | Full drafting | Requirements, constraints, review/acceptance |
| Metrics reporting | Populated the README table from `outputs/metrics.json` | Verified numbers are real, not invented |

## Guardrails the human imposed

- **No invented numbers.** The README metrics table is copied from the real
  `outputs/metrics.json` produced by an actual run.
- **Prevalence honesty.** The ~2% sample prevalence (all positives kept) versus
  the true ~0.10% is disclosed in the README; precision figures are flagged as
  optimistic.
- **No committed secrets.** `ANTHROPIC_API_KEY` and Kaggle credentials are read
  from environment variables only; `.gitignore` excludes `.env` and data.
- **Local vs. clean-target separation.** Intel/Rosetta workarounds
  (`torch==2.2.2`, `polars[rtcompat]`, LightGBM subprocess isolation) stay local;
  Docker and CI target clean `linux/amd64` with the standard pins.

## The wrapper prompt (artifact)

The industrial-repo wrapper (this docs file, the Docker/compose setup, the
FastAPI service, DVC + MLflow scaffolding, Makefile, pre-commit, the enhanced CI
and the README) was produced from a single task brief given to Claude Code. Its
essence:

> Wrap an existing, working MVP (`AML-GRAPH`, explainable money-laundering
> detection via graph motifs) in an industrial-looking repository for the
> "Engineering & Development" criterion of a Junior ML contest. Produce complete
> files on disk. **Do not run any git commands** — another process handles git.
> **Do not hardcode secrets** — reference the Kaggle token and
> `ANTHROPIC_API_KEY` via environment variables only. The local dev machine is
> an Intel Homebrew Python under Rosetta needing `torch==2.2.2`,
> `polars[rtcompat]` and LightGBM-in-subprocess to avoid an OpenMP clash — these
> are **local-only** workarounds and must **not** be baked into Docker/CI, which
> target clean `linux/amd64`.
>
> Create: a reproducible CPU `Dockerfile` (libgomp1, non-root, commented CUDA
> option, Streamlit CMD); a `docker-compose.yml` with `demo` and `api` services
> reading secrets from host env; `app/api.py` (FastAPI `POST /score_subgraph` +
> `GET /health`, mirroring the Streamlit artifact loading and reusing
> `src.narrative`); a Hydra-style `configs/config.yaml` mirroring `RunConfig`
> plus an optional `src/config_loader.py`; a `dvc.yaml` pipeline
> (download -> build_graph -> train_baseline -> train_gnn -> figures) with a
> local remote stub and a `data/` note that data is DVC/Kaggle-pulled, never
> committed; an MLflow `src/tracking.py` helper wired optionally into
> `pipeline.run_mvp(track=False)`; a professional `README.md` with a mermaid
> architecture diagram and a metrics table populated from the **real**
> `outputs/metrics.json` (with the ~2% vs 0.10% prevalence disclosure); a
> `Makefile` (setup/lint/test/experiment/demo/api/figures/clean); a pinned
> `.pre-commit-config.yaml` (ruff + hygiene hooks); an enhanced CI with a
> ruff+pytest job and a buildx docker-build job; and this `docs/ai_usage.md`.
> Everything ruff-clean (line-length 100), type-hinted and docstringed.

## Reproducing the AI-assisted workflow

The verification the agent ran on its own output:

```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/python -m py_compile app/api.py src/tracking.py src/config_loader.py
```

AI accelerated the *engineering surface area*; the modelling decisions, honesty
constraints and final acceptance remained human.
