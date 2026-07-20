"""
Download the real Kaggle datasets for all four trained models into ai-engine/data/.

Prerequisites
-------------
1. A free Kaggle account.
2. An API token: kaggle.com -> your avatar -> Settings -> "Create New Token".
   This downloads `kaggle.json`. Put it at:
     Windows: C:\\Users\\<you>\\.kaggle\\kaggle.json
     macOS/Linux: ~/.kaggle/kaggle.json
   (or set KAGGLE_USERNAME / KAGGLE_KEY env vars.)
3. Install the client:  pip install kaggle   (already in requirements.txt)

Run
---
    python -m scripts.fetch_datasets            # fetch all four
    python -m scripts.fetch_datasets lead churn # fetch only some

Each dataset is downloaded, unzipped, the right CSV is picked and copied to the
filename the matching trainer expects. Then retrain with:
    python -m scripts.train_all
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# key -> (kaggle dataset slug, target filename in data/, substring to prefer when
#         the download contains multiple CSVs)
DATASETS = {
    "lead":   ("amritachatterjee09/lead-scoring-dataset",                      "Lead Scoring.csv",           "lead"),
    "churn":  ("blastchar/telco-customer-churn",                               "Telco-Customer-Churn.csv",   "telco"),
    "sentiment": ("mansithummar67/171k-product-review-with-sentiment-dataset", "reviews.csv",                "review"),
    "recommend": ("vijayuv/onlineretail",                                      "retail.csv",                 "retail"),
}


def _kaggle_cmd() -> list[str]:
    """Locate the kaggle CLI that ships with the current interpreter's venv."""
    scripts_dir = Path(sys.executable).parent
    for name in ("kaggle.exe", "kaggle"):
        candidate = scripts_dir / name
        if candidate.exists():
            return [str(candidate)]
    return ["kaggle"]  # fall back to PATH


def _pick_csv(folder: Path, prefer: str) -> Path | None:
    csvs = list(folder.rglob("*.csv"))
    if not csvs:
        return None
    preferred = [c for c in csvs if prefer.lower() in c.name.lower()]
    pool = preferred or csvs
    # Largest file in the chosen pool (the main data table, not a small lookup).
    return max(pool, key=lambda p: p.stat().st_size)


def fetch(key: str) -> bool:
    slug, target, prefer = DATASETS[key]
    print(f"\n[{key}] downloading {slug} ...")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        result = subprocess.run(
            _kaggle_cmd() + ["datasets", "download", "-d", slug, "-p", str(tmp_path), "--unzip"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"[{key}] kaggle download failed:\n{result.stderr.strip() or result.stdout.strip()}")
            return False
        csv = _pick_csv(tmp_path, prefer)
        if not csv:
            print(f"[{key}] ERROR: no CSV found in the download.")
            return False
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        dest = DATA_DIR / target
        shutil.copyfile(csv, dest)
        print(f"[{key}] saved {csv.name} -> {dest.name}  ({dest.stat().st_size // 1024} KB)")
        return True


def main(argv: list[str]) -> None:
    keys = argv or list(DATASETS.keys())
    unknown = [k for k in keys if k not in DATASETS]
    if unknown:
        sys.exit(f"Unknown dataset key(s): {unknown}. Valid: {list(DATASETS)}")

    ok = sum(fetch(k) for k in keys)
    print(f"\n{ok}/{len(keys)} datasets fetched into {DATA_DIR}")
    if ok:
        print("Retrain the models with:  python -m scripts.train_all")


if __name__ == "__main__":
    main(sys.argv[1:])
