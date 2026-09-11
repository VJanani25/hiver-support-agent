# Data notes

`sample/support_sample.csv` is a small, deterministic demo fixture created for smoke tests. It is not the Kaggle benchmark and should not be used to claim production or benchmark performance.

The full Customer Support on Twitter export is intentionally excluded because of size and dataset licensing. Use `scripts/download_data.py` for setup instructions, then keep the raw file under `data/raw/`, which is ignored by Git.

`golden_set.csv` is a candidate template. Rows with blank `intent` and `expected_action` require human review. Never treat `suggested_intent` as a gold label.