# Reproduction

Use the existing Anaconda Python environment with NumPy, SciPy, sklearn, Torch and joblib. Preserve historical dependencies listed in PLAN.json.

1. PLAN.json / PROTOCOL.md freeze the candidate matrix; do not regenerate them.
2. `python -B launch.py` resumes only missing model contexts; no evaluation-dependent branches.
3. `python -B analyze.py score` exports predictions and frozen actions before reference joins.
4. `python -B analyze.py summarize` creates statistical tables.
5. `python -B finalize.py` verifies and inventories.
6. `python -B render.py` makes SVG/PDF/PNG appendix figures.

No patient data are uploaded. Record-level tables are local private source data, not automatically public-shareable. Copying this folder alone does not grant permission to distribute raw data.
