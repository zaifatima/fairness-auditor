# Proactive Fairness Auditor for Recommender Systems

MSc dissertation project (Middlesex University Dubai, supervised by Dr.
Krishnadasn Nanath): a toolkit for measuring fairness drift in
recommender systems over time, rather than only auditing a single
snapshot, and forecasting future fairness-threshold violations before
they occur.

## What's actually built vs. planned

This repo reflects a project that has spent most of its effort so far
on getting a working, resilient training pipeline running on
constrained infrastructure — several of the files here exist
specifically because an earlier, simpler approach failed under real
conditions (forced cloud reboots, memory limits) and had to be
redesigned. That history is deliberately not hidden; see
`INFRASTRUCTURE.md` for the full account.

**Done:**
- Data conversion pipeline for MovieLens-32M and LastFM-1K
- Three recommender backbones trained via RecBole: NCF, LightGCN, SASRec
  (SASRec still finishing its final epochs at time of writing)
- A resumable training script that survives repeated cloud-instance
  reboots with at most one epoch of lost progress
- ML-1M demographic data (age/gender/occupation) prepared locally, plus
  per-user profile text, ready for an LLM-based fourth recommender

**Planned / in progress, not yet built:**
- `generate_recommendations.py` — will populate the shared
  `recommendations` table from each trained model's checkpoint
- The fairness metrics engine (demographic parity gap, exposure ratio,
  coverage, KS-statistic, rND, rKL, PRAG) — designed, not yet coded
- The forecasting module (Random Forest predicting future fairness
  violations)
- The LLM-as-recommender pilot (data is ready; API-calling script is
  the next step)

## Repository structure

### Core pipeline
| File | Purpose |
|---|---|
| `convert_movielens.py` | Converts raw MovieLens-32M CSVs to RecBole atomic format |
| `convert_lastfm.py` | Converts raw Last.fm TSV to RecBole atomic format (artist-level items — see dissertation Ch.4 for why) |
| `make_medium_sample.py`, `make_lightgcn_sample.py` | Build the 50k/5k-user subsamples used for hardware-constrained training |
| `ncf.yaml`, `lightgcn.yaml`, `sasrec.yaml` | Per-model RecBole configs — comments/values reflect real hardware-driven tuning, not defaults |
| `train.py` | **The most important file in this repo.** Takes manual control of RecBole's training loop (rather than using its built-in shortcut) specifically to support crash-safe resume — see `INFRASTRUCTURE.md` |
| `run_training.sh` | Auto-detects the latest checkpoint and resumes, or starts fresh |

### Evaluation
| File | Purpose |
|---|---|
| `evaluate_lightgcn_local.py` | Runs full-catalogue-scale evaluation locally (not on the cloud instance), because LightGCN's evaluation cost couldn't survive the cloud's forced reboot cycle |

### LLM extension (in progress)
| File | Purpose |
|---|---|
| `prepare_ml1m.py` | Downloads/parses ML-1M (the one MovieLens release with real user demographics) and builds per-user profile text |
| `build_llm_candidates.py` | Builds uni100-style test candidate sets per user, for direct comparability with the other three models |

### Documentation
| File | Purpose |
|---|---|
| `INFRASTRUCTURE.md` | The full account of the AWS lab-account constraints hit, and exactly how the resume/checkpoint/cron system was built to survive them |

## Why some results are missing or partial

This is stated here plainly rather than left for a reader to wonder
about: SASRec's training has taken substantially longer than NCF or
LightGCN's did, because of its Transformer attention cost — full
details and the fixes attempted are in the dissertation's Analysis and
Design and Results chapters, not just this README.
