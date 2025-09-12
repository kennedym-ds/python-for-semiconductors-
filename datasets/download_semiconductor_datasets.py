"""Utilities to help users acquire open semiconductor and manufacturing-related datasets.

NOTE: For datasets requiring authentication (Kaggle, IEEE DataPort), this script
provides instructions and placeholders but will not bypass required logins.

Implemented:
- SECOM dataset (UCI Machine Learning Repository) automatic download
- Steel Plates Faults dataset (UCI Machine Learning Repository)
- Modern UCI ML repository support via ucimlrepo package

Placeholders:
- WM-811K (Kaggle) -> user must use Kaggle API
- Wafer surface / mixed-type defect datasets (IEEE DataPort)

Usage (PowerShell):
  python datasets/download_semiconductor_datasets.py --dataset secom
  python datasets/download_semiconductor_datasets.py --dataset steel-plates
  python datasets/download_semiconductor_datasets.py --dataset all

"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import textwrap
from pathlib import Path
from urllib.request import urlretrieve

SECOM_FILES = {
    "secom.data": "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data",
    "secom_labels.data": "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data",
    "secom.names": "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.names",
}

# Steel Plates Faults dataset info
STEEL_PLATES_UCI_ID = 198

# Optional published SHA256 checksums (populate if you curate internally)
SECOM_SHA256: dict[str, str] = {}


def check_ucimlrepo() -> bool:
    """Check if ucimlrepo package is available for modern UCI downloads."""
    try:
        import ucimlrepo  # noqa: F401

        return True
    except ImportError:
        return False


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_secom(root: Path) -> None:
    secom_dir = root / "secom"
    secom_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading SECOM dataset into {secom_dir} ...")

    # Create README for SECOM dataset
    readme_content = textwrap.dedent(
        """
    # SECOM Dataset
    
    **Source**: UCI Machine Learning Repository  
    **Domain**: Semiconductor Manufacturing Process Control  
    **Task**: Binary Classification (Pass/Fail prediction)
    
    ## Description
    
    The SECOM dataset contains semiconductor manufacturing data collected from a semi-conductor fabrication process. Each row represents a sensor measurement from the manufacturing process, with 590 features representing various process parameters and measurements.
    
    ## Files
    
    - `secom.data`: Feature matrix (1567 instances × 590 features)
    - `secom_labels.data`: Binary target labels (-1 = Pass, +1 = Fail)  
    - `secom.names`: Dataset description and attribute information
    
    ## Usage Notes
    
    - High dimensionality: 590 features for quality control sensors
    - Class imbalance: Most wafers pass (majority class -1)
    - Missing values: Some sensors may have null readings
    - Real semiconductor fab data with anonymized feature names
    
    ## Citation
    
    ```
    Dua, D. and Graff, C. (2019). UCI Machine Learning Repository
    [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California,
    School of Information and Computer Science.
    ```
    
    Original source: https://archive.ics.uci.edu/ml/datasets/SECOM
    """
    ).strip()

    readme_path = secom_dir / "README.md"
    if not readme_path.exists():
        readme_path.write_text(readme_content, encoding="utf-8")
        print(f"  [created] SECOM README.md")

    for fname, url in SECOM_FILES.items():
        dest = secom_dir / fname
        if dest.exists():
            print(f"  [skip] {fname} already exists")
            continue
        print(f"  [get ] {fname} -> {url}")
        urlretrieve(url, dest)
        if fname in SECOM_SHA256:
            digest = sha256sum(dest)
            if digest != SECOM_SHA256[fname]:
                raise ValueError(
                    f"Checksum mismatch for {fname} (expected {SECOM_SHA256[fname]}, got {digest})"
                )
    print("SECOM download complete.")


def download_steel_plates(root: Path) -> None:
    """Download Steel Plates Faults dataset using UCI ML repo API or fallback."""
    steel_dir = root / "steel-plates"
    steel_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Steel Plates Faults dataset into {steel_dir} ...")

    # Create README for Steel Plates dataset
    readme_content = textwrap.dedent(
        """
    # Steel Plates Faults Dataset
    
    **Source**: UCI Machine Learning Repository (ID: 198)  
    **Domain**: Manufacturing Quality Control  
    **Task**: Multi-class Classification (7 fault types)
    
    ## Description
    
    A dataset of steel plates' faults, classified into 7 different types. The goal was to train machine learning for automatic pattern recognition in steel manufacturing.
    
    ## Fault Types
    
    1. **Pastry** - Surface texture defects
    2. **Z_Scratch** - Longitudinal scratches  
    3. **K_Scratch** - Transverse scratches
    4. **Stains** - Discoloration defects
    5. **Dirtiness** - Contamination
    6. **Bumps** - Surface elevation defects
    7. **Other_Faults** - Miscellaneous defects
    
    ## Dataset Characteristics
    
    - **Instances**: 1,941 steel plates
    - **Features**: 27 geometric and luminosity measurements
    - **Missing Values**: None
    - **Feature Types**: Integer and Real
    
    ## Key Features
    
    - Geometric measurements (X/Y min/max, areas, perimeters)
    - Luminosity statistics (sum, min, max, luminosity index)
    - Steel properties (type, thickness)
    - Shape indices (edges, orientation, sigmoid transforms)
    
    ## Citation
    
    ```
    Buscema, M., Terzi, S., & Tastle, W. (2010). Steel Plates Faults [Dataset]. 
    UCI Machine Learning Repository. https://doi.org/10.24432/C5J88N
    ```
    
    Original source: https://archive.ics.uci.edu/dataset/198/steel+plates+faults
    """
    ).strip()

    readme_path = steel_dir / "README.md"
    if not readme_path.exists():
        readme_path.write_text(readme_content, encoding="utf-8")
        print(f"  [created] Steel Plates README.md")

    # Try modern UCI ML repo approach first
    if check_ucimlrepo():
        try:
            from ucimlrepo import fetch_ucirepo

            print(
                f"  [fetch] Using ucimlrepo package for dataset ID {STEEL_PLATES_UCI_ID}"
            )
            dataset = fetch_ucirepo(id=STEEL_PLATES_UCI_ID)

            # Save as CSV files for easier use
            features_path = steel_dir / "steel_plates_features.csv"
            targets_path = steel_dir / "steel_plates_targets.csv"

            if not features_path.exists():
                dataset.data.features.to_csv(features_path, index=False)
                print(f"  [saved] Features -> {features_path.name}")

            if not targets_path.exists():
                dataset.data.targets.to_csv(targets_path, index=False)
                print(f"  [saved] Targets -> {targets_path.name}")

            # Save metadata
            metadata_path = steel_dir / "metadata.txt"
            if not metadata_path.exists():
                with metadata_path.open("w", encoding="utf-8") as f:
                    f.write("=== DATASET METADATA ===\n")
                    f.write(str(dataset.metadata))
                    f.write("\n\n=== VARIABLE INFORMATION ===\n")
                    f.write(str(dataset.variables))
                print(f"  [saved] Metadata -> {metadata_path.name}")

        except Exception as e:
            print(f"  [error] UCI ML repo fetch failed: {e}")
            print(f"  [fallback] Manual download not implemented for Steel Plates")
            return
    else:
        print(f"  [info] ucimlrepo not available. Install with: pip install ucimlrepo")
        print(f"  [skip] Steel Plates dataset requires ucimlrepo package")
        return

    print("Steel Plates download complete.")


def kaggle_instructions(dataset: str) -> str:
    """Generate instructions for downloading Kaggle datasets."""
    return textwrap.dedent(
        f"""
    Dataset '{dataset}' requires Kaggle authentication.
    Steps:
      1. Install Kaggle CLI: pip install kaggle
      2. Obtain API token (Account > Create New Token) and place kaggle.json in %USERPROFILE%/.kaggle (Windows) or ~/.kaggle
      3. Run: kaggle datasets download -d {dataset} -p datasets/wm811k/raw --unzip
      4. Do NOT commit large extracted files.
    """
    )


def download_all_datasets(root: Path) -> None:
    """Download all supported datasets."""
    print("Downloading all supported datasets...")
    download_secom(root)
    print()  # spacing
    download_steel_plates(root)
    print()  # spacing
    download_wm811k_placeholder(root / "wm811k")
    print("All dataset downloads completed.")


def download_wm811k_placeholder(root: Path) -> None:
    """Create placeholder instructions for WM-811K Kaggle dataset."""
    root.mkdir(parents=True, exist_ok=True)
    print(kaggle_instructions("qingyi/wm811k-wafer-map"))
    (root / "README_fetch.txt").write_text(
        "See console instructions for Kaggle download process.\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download semiconductor and manufacturing datasets"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["secom", "steel-plates", "wm811k", "all"],
        help="Dataset key to download or scaffold",
    )
    parser.add_argument(
        "--datasets-dir", default="datasets", help="Base datasets directory"
    )
    args = parser.parse_args(argv)

    base = Path(args.datasets_dir)

    if args.dataset == "secom":
        download_secom(base)
    elif args.dataset == "steel-plates":
        download_steel_plates(base)
    elif args.dataset == "wm811k":
        download_wm811k_placeholder(base / "wm811k")
    elif args.dataset == "all":
        download_all_datasets(base)
    else:
        parser.error("Unsupported dataset key")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
