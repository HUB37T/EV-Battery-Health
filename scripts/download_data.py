"""Download the Kaggle archive into data/raw/ and verify the audited CSV hash.

python scripts/download_data.py           # skip if the audited files are already present
python scripts/download_data.py --force   # download again

The dataset is public, so no Kaggle account or token is needed. Only the three
expected files are extracted. The CSV must match the SHA-256 the audit was run on.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
URL = (
    "https://www.kaggle.com/api/v1/datasets/download/vinay24baghira/ev-battery-health-and-degradation-dataset"
)
FILES = ("ev_battery_health.csv", "README.md", "data_dictionary .csv")
CSV_SHA256 = "cacdf6afcf50db86583be3754207a89acbfd60b04dfff8f69b71e29cfd78218d"


def csv_matches() -> bool:
    path = RAW / FILES[0]
    return path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == CSV_SHA256


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--force", action="store_true", help="download even if the files are present")
    args = parser.parse_args()
    if not args.force and csv_matches() and all((RAW / name).is_file() for name in FILES):
        print("data/raw already contains the audited files, nothing to do.")
        return
    print(f"Downloading {URL}")
    request = urllib.request.Request(URL, headers={"User-Agent": "ev-battery-health-download"})
    with urllib.request.urlopen(request, timeout=120) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    missing = [name for name in FILES if name not in archive.namelist()]
    if missing:
        sys.exit(f"The archive does not contain {missing}. Download it manually, see data/README.md.")
    RAW.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        (RAW / name).write_bytes(archive.read(name))
    if not csv_matches():
        sys.exit("The downloaded CSV differs from the audited version. Do not run the pipeline on it.")
    print("Saved " + ", ".join(FILES) + " to data/raw/ and verified the CSV hash.")


if __name__ == "__main__":
    main()
