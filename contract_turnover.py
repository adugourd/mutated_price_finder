#!/usr/bin/env python3
"""
EVE Online Contract Turnover Analyzer

Analyzes EVE Ref public contract snapshots to determine which mutated modules
have the highest turnover (demand) based on contracts that disappear between
daily snapshots.

A contract that disappears could be:
- Sold (what we're trying to measure)
- Cancelled by seller
- Expired (we filter these out using date_expired)

Over a 2-month period, the noise from cancellations should average out,
giving us a reasonable proxy for relative demand.
"""

from __future__ import annotations

import re
import tarfile
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import TypedDict

import requests
import pandas as pd


class SnapshotData(TypedDict):
    """Data loaded from a contract snapshot."""
    contracts: pd.DataFrame
    dynamic_items: pd.DataFrame

# Cache directory
CACHE_DIR = Path(__file__).parent / ".cache" / "contract_history"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# EVE Ref URLs
EVEREF_HISTORY_URL = "https://data.everef.net/public-contracts/history"
FUZZWORK_SDE = "https://www.fuzzwork.co.uk/dump/latest"

HEADERS = {
    "User-Agent": "EVE Contract Turnover Analyzer - github.com/mutated-price-finder",
}


def get_snapshot_url(date: datetime) -> str:
    """Get URL for the first snapshot of a given date."""
    date_str = date.strftime("%Y-%m-%d")
    year = date.strftime("%Y")
    # Use the midnight snapshot (00:00:08 pattern from observed data)
    filename = f"public-contracts-{date_str}_00-00-*.v2.tar.bz2"
    return f"{EVEREF_HISTORY_URL}/{year}/{date_str}/"


def download_snapshot(date: datetime) -> Path:
    """Download and cache a daily snapshot."""
    date_str = date.strftime("%Y-%m-%d")
    cache_path = CACHE_DIR / f"{date_str}.tar.bz2"

    if cache_path.exists():
        return cache_path

    # List directory to find the first snapshot of the day
    year = date.strftime("%Y")
    dir_url = f"{EVEREF_HISTORY_URL}/{year}/{date_str}/"

    print(f"  Fetching directory listing for {date_str}...")
    response = requests.get(dir_url, headers=HEADERS, timeout=30)
    response.raise_for_status()

    # Parse HTML to find first .tar.bz2 file
    # Links are in format: href="/public-contracts/history/2025/2025-12-19/public-contracts-2025-12-19_00-00-08.v2.tar.bz2"
    content = response.text
    pattern = rf'href="(/public-contracts/history/{year}/{date_str}/public-contracts-{date_str}_[^"]+\.tar\.bz2)"'
    match = re.search(pattern, content)

    if not match:
        raise ValueError(f"No snapshot found for {date_str}")

    file_path = match.group(1)
    filename = file_path.split('/')[-1]

    # Download the snapshot
    file_url = f"https://data.everef.net{file_path}"
    print(f"  Downloading {filename}...")

    response = requests.get(file_url, headers=HEADERS, timeout=120, stream=True)
    response.raise_for_status()

    with open(cache_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return cache_path


def extract_csv_from_archive(archive_path: Path, csv_name: str) -> pd.DataFrame:
    """Extract a specific CSV file from the tar.bz2 archive."""
    with tarfile.open(archive_path, 'r:bz2') as tar:
        for member in tar.getmembers():
            if member.name.endswith(csv_name):
                f = tar.extractfile(member)
                if f:
                    return pd.read_csv(f)
    raise FileNotFoundError(f"Could not find {csv_name} in archive")


def load_snapshot_data(archive_path: Path) -> SnapshotData:
    """Load contracts and dynamic items from a snapshot."""
    data: SnapshotData = {}  # type: ignore[typeddict-item]

    # Load contracts (need contract_id, date_expired, price)
    contracts = extract_csv_from_archive(archive_path, 'contracts.csv')
    data['contracts'] = contracts[['contract_id', 'date_expired', 'price', 'type']].copy()

    # Load dynamic items (mutated modules with source_type_id and mutator_type_id)
    dynamic_items = extract_csv_from_archive(archive_path, 'contract_dynamic_items.csv')
    data['dynamic_items'] = dynamic_items[['contract_id', 'item_id', 'source_type_id', 'mutator_type_id']].copy()

    return data


def load_type_names() -> dict[int, str]:
    """Load type names from SDE."""
    cache_path = CACHE_DIR.parent / "invTypes.csv"

    if not cache_path.exists():
        print("  Downloading type names from Fuzzwork SDE...")
        response = requests.get(f"{FUZZWORK_SDE}/invTypes.csv", headers=HEADERS, timeout=60)
        response.raise_for_status()
        cache_path.write_bytes(response.content)

    types_df = pd.read_csv(cache_path)
    return dict(zip(types_df['typeID'], types_df['typeName']))


def analyze_turnover(dates: list[datetime], type_names: dict[int, str]) -> pd.DataFrame:
    """Analyze contract turnover across multiple days."""

    all_disappeared = []

    prev_data = None
    prev_date = None

    for date in sorted(dates):
        print(f"\nProcessing {date.strftime('%Y-%m-%d')}...")

        try:
            archive_path = download_snapshot(date)
            current_data = load_snapshot_data(archive_path)
        except Exception as e:
            print(f"  Error: {e}")
            continue

        if prev_data is not None:
            # Find contracts that disappeared
            prev_contract_ids = set(prev_data['contracts']['contract_id'])
            curr_contract_ids = set(current_data['contracts']['contract_id'])

            disappeared_ids = prev_contract_ids - curr_contract_ids

            if disappeared_ids:
                # Get details of disappeared contracts
                disappeared_contracts = prev_data['contracts'][
                    prev_data['contracts']['contract_id'].isin(disappeared_ids)
                ].copy()

                # Filter out expired contracts
                disappeared_contracts['date_expired'] = pd.to_datetime(
                    disappeared_contracts['date_expired'], errors='coerce', utc=True
                )
                current_time = pd.Timestamp(date, tz='UTC')

                # Keep only contracts that hadn't expired yet (likely sold or cancelled)
                not_expired = disappeared_contracts[
                    disappeared_contracts['date_expired'] > current_time
                ]

                # Join with dynamic items to get source_type_id and mutator_type_id
                dynamic_items = prev_data['dynamic_items']
                disappeared_with_details = not_expired.merge(
                    dynamic_items, on='contract_id', how='inner'
                )

                if not disappeared_with_details.empty:
                    disappeared_with_details['disappeared_date'] = date
                    disappeared_with_details['prev_date'] = prev_date
                    all_disappeared.append(disappeared_with_details)

                print(f"  Found {len(disappeared_ids)} disappeared contracts")
                print(f"  After filtering expired: {len(not_expired)}")
                print(f"  With mutated modules: {len(disappeared_with_details)}")

        prev_data = current_data
        prev_date = date

    if not all_disappeared:
        return pd.DataFrame()

    # Combine all disappeared contracts
    df = pd.concat(all_disappeared, ignore_index=True)

    # Add type names
    df['source_name'] = df['source_type_id'].map(type_names).fillna('Unknown')
    df['mutator_name'] = df['mutator_type_id'].map(type_names).fillna('Unknown')

    return df


def format_isk(value: float) -> str:
    """Format ISK value with suffix."""
    if value >= 1_000_000_000:
        return f"{value/1_000_000_000:.2f}B"
    elif value >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    elif value >= 1_000:
        return f"{value/1_000:.1f}K"
    return f"{value:.0f}"


def main() -> None:
    """Main entry point for the contract turnover analyzer CLI."""
    parser = argparse.ArgumentParser(
        description="Analyze contract turnover for mutated modules"
    )
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=60,
        help="Number of days to analyze (default: 60)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output CSV file for detailed results"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("EVE Online Contract Turnover Analyzer")
    print("=" * 70)
    print(f"\nAnalyzing {args.days} days of contract data...")
    print("(Contracts that disappear before expiry = likely sold)")

    # Generate list of dates to analyze
    end_date = datetime.now() - timedelta(days=1)  # Yesterday (today might be incomplete)
    dates = [end_date - timedelta(days=i) for i in range(args.days)]
    dates = sorted(dates)

    print(f"\nDate range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")

    # Load type names
    print("\nLoading SDE data...")
    type_names = load_type_names()

    # Analyze turnover
    print("\n" + "=" * 70)
    print("DOWNLOADING AND ANALYZING SNAPSHOTS")
    print("=" * 70)

    df = analyze_turnover(dates, type_names)

    if df.empty:
        print("\nNo turnover data found!")
        return

    # Results
    print("\n" + "=" * 70)
    print("TURNOVER RESULTS")
    print("=" * 70)

    print(f"\nTotal mutated module contracts that disappeared: {len(df)}")
    print(f"Unique contracts: {df['contract_id'].nunique()}")

    # Top source modules by turnover
    print("\n" + "-" * 60)
    print("TOP SOURCE MODULES (base modules with highest turnover)")
    print("-" * 60)
    source_counts = df.groupby('source_name').agg({
        'contract_id': 'nunique',
        'price': 'mean'
    }).rename(columns={'contract_id': 'count', 'price': 'avg_price'})
    source_counts = source_counts.sort_values('count', ascending=False)

    for name, row in source_counts.head(20).iterrows():
        avg_price = format_isk(row['avg_price'])
        print(f"  {int(row['count']):>4}x  {name:<40} (avg: {avg_price})")

    # Top mutaplasmids by turnover
    print("\n" + "-" * 60)
    print("TOP MUTAPLASMIDS (mutation types with highest turnover)")
    print("-" * 60)
    mutator_counts = df.groupby('mutator_name').agg({
        'contract_id': 'nunique',
        'price': 'mean'
    }).rename(columns={'contract_id': 'count', 'price': 'avg_price'})
    mutator_counts = mutator_counts.sort_values('count', ascending=False)

    for name, row in mutator_counts.head(20).iterrows():
        avg_price = format_isk(row['avg_price'])
        print(f"  {int(row['count']):>4}x  {name:<45} (avg: {avg_price})")

    # Top combinations
    print("\n" + "-" * 60)
    print("TOP SOURCE + MUTAPLASMID COMBINATIONS")
    print("-" * 60)
    combo_counts = df.groupby(['source_name', 'mutator_name']).agg({
        'contract_id': 'nunique',
        'price': 'mean'
    }).rename(columns={'contract_id': 'count', 'price': 'avg_price'})
    combo_counts = combo_counts.sort_values('count', ascending=False)

    for (source, mutator), row in combo_counts.head(25).iterrows():
        avg_price = format_isk(row['avg_price'])
        combo_name = f"{source} + {mutator}"
        if len(combo_name) > 65:
            combo_name = combo_name[:62] + "..."
        print(f"  {int(row['count']):>4}x  {combo_name:<65} ({avg_price})")

    # Daily turnover rate
    print("\n" + "-" * 60)
    print("TURNOVER STATISTICS")
    print("-" * 60)
    daily_avg = len(df) / args.days
    print(f"  Average contracts/day: {daily_avg:.1f}")
    print(f"  Total ISK volume: {format_isk(df['price'].sum())}")
    print(f"  Average price: {format_isk(df['price'].mean())}")
    print(f"  Median price: {format_isk(df['price'].median())}")

    # Save to CSV if requested
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
