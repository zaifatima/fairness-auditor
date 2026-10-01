# Data and generated artefacts

Large raw and derived datasets are intentionally excluded from version control.

This repository is intended to contain the source code, configuration files, fairness-auditing logic, temporal evaluation scripts, small aggregate outputs, and dashboard components for the MSc project **Proactive Fairness Auditing in Recommender Systems**.

Excluded from GitHub because they are large, generated, sensitive, or reproducible:

- raw MovieLens / Last.fm downloads;
- RecBole-formatted full datasets;
- yearly temporal train/test datasets;
- large intermediate interaction files;
- trained checkpoint directories;
- TensorBoard and training logs;
- local SQLite databases;
- API keys, PEM files, and other credentials.

The large data artefacts can be recreated from the preparation and temporal-pipeline scripts in the repository. Small aggregate CSV outputs that support the reported thesis results or dashboard visualisation may remain in the repository.

Before making the repository public, verify that no credentials have ever been committed to Git history.
