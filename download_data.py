"""
COSMO — Download datasets
  - HYG Database v3 (119k stelle con RA/Dec/magnitudine)
  - Kepler KOI catalog (oggetti di interesse con probabilità pianeta)
  - Constellation lines (Stellarium format)
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

DOWNLOADS = {
    "hyg_database.csv": (
        "https://raw.githubusercontent.com/astronexus/HYG-Database/main/hygdata_v3.csv",
        "HYG star catalog v3 (~119k stelle)"
    ),
    "kepler_koi.csv": (
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+cumulative&format=csv",
        "Kepler KOI cumulative catalog"
    ),
}


def download(url: str, dest: Path, label: str):
    if dest.exists():
        print(f"  [skip] {dest.name} già presente")
        return
    print(f"  Downloading {label}...")
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
    print(f"  Salvato: {dest}")


if __name__ == "__main__":
    print("=== COSMO: download dati ===")
    for filename, (url, label) in DOWNLOADS.items():
        download(url, DATA_DIR / filename, label)
    print("\nDone. Dati in:", DATA_DIR)
