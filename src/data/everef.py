"""
EVE Ref contract data fetching and parsing.

Downloads and extracts contract snapshots from EVE Ref public contract data.
"""

import tarfile
from pathlib import Path
from datetime import datetime
from typing import Optional

import requests
import pandas as pd

from src.config.loader import load_constants

# Load configuration
_constants = load_constants()
EVEREF_CONTRACTS_URL = _constants['api']['everef_contracts_url']
CACHE_DURATION = _constants['cache']['contract_cache_duration']


def get_cache_dir() -> Path:
    """Get the cache directory path, creating it if needed."""
    cache_dir = Path(__file__).parent.parent.parent / ".cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


def download_contract_archive(cache_dir: Optional[Path] = None) -> Path:
    """
    Download contract data archive if not cached or cache expired.

    Args:
        cache_dir: Optional custom cache directory

    Returns:
        Path to the downloaded archive
    """
    if cache_dir is None:
        cache_dir = get_cache_dir()

    archive_path = cache_dir / "public-contracts-latest.v2.tar.bz2"

    # Check if cache is still valid
    if archive_path.exists():
        age_seconds = datetime.now().timestamp() - archive_path.stat().st_mtime
        if age_seconds < CACHE_DURATION:
            return archive_path

    print("Downloading contract data...")
    response = requests.get(EVEREF_CONTRACTS_URL, stream=True, timeout=120)
    response.raise_for_status()

    with open(archive_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return archive_path


def extract_csv_from_archive(archive_path: Path, csv_name: str) -> pd.DataFrame:
    """
    Extract a specific CSV file from the contract archive.

    Args:
        archive_path: Path to the tar.bz2 archive
        csv_name: Name of the CSV file to extract (e.g., 'contracts.csv')

    Returns:
        DataFrame with the CSV contents

    Raises:
        FileNotFoundError: If the CSV file is not found in the archive
    """
    with tarfile.open(archive_path, 'r:bz2') as tar:
        for member in tar.getmembers():
            if member.name.endswith(csv_name):
                f = tar.extractfile(member)
                if f:
                    return pd.read_csv(f)
    raise FileNotFoundError(f"Could not find {csv_name} in archive")


def get_contracts_with_items(archive_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get contracts and dynamic items DataFrames from archive.

    Args:
        archive_path: Path to the contract archive

    Returns:
        Tuple of (contracts_df, dynamic_items_df)
    """
    contracts = extract_csv_from_archive(archive_path, 'contracts.csv')
    dynamic_items = extract_csv_from_archive(archive_path, 'contract_dynamic_items.csv')
    return contracts, dynamic_items


def get_dogma_attributes(archive_path: Path) -> pd.DataFrame:
    """
    Get dogma attributes for dynamic items from archive.

    Args:
        archive_path: Path to the contract archive

    Returns:
        DataFrame with dogma attributes
    """
    return extract_csv_from_archive(archive_path, 'contract_dynamic_items_dogma_attributes.csv')
