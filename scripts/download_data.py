from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Print reproducible Kaggle download instructions.")
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    print("Dataset: thoughtvector/customer-support-on-twitter")
    print("Download using Kaggle's documented CLI or browser flow:")
    print("  kaggle datasets download -d thoughtvector/customer-support-on-twitter -p", args.output)
    print("Then unzip the export and pass its CSV to scripts/prepare_data.py.")
    print("Raw data stays outside Git because it is large and subject to dataset terms.")


if __name__ == "__main__":
    main()