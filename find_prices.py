#!/usr/bin/env python3
"""
EVE Online Mutated Module Price Finder

Finds the lowest priced contracts for mutated modules with equivalent or worse stats.
Uses EVE Ref public contract data which includes mutated item dogma attributes.

Sources:
- EVE Ref Public Contracts: https://data.everef.net/public-contracts/
- Docs: https://docs.everef.net/datasets/public-contracts.html
"""

from __future__ import annotations

import tarfile
import tempfile
import argparse
from pathlib import Path
from datetime import datetime
from typing import TypedDict

import requests
import pandas as pd


class ModuleTypeConfig(TypedDict):
    """Configuration for a module type."""
    name: str
    source_type_ids: set[int]
    type_names: dict[int, str]


class ContractData(TypedDict):
    """Data loaded from contract archives."""
    contracts: pd.DataFrame
    items: pd.DataFrame
    dynamic_items: pd.DataFrame
    dogma_attributes: pd.DataFrame

# EVE Ref data URL
EVEREF_CONTRACTS_URL = "https://data.everef.net/public-contracts/public-contracts-latest.v2.tar.bz2"

# Dogma Attribute IDs (from EVE SDE)
ATTR_DAMAGE_MODIFIER = 64       # damageMultiplier
ATTR_ROF_BONUS = 204            # speedMultiplier (rate of fire)
ATTR_CPU = 50                   # cpu

# Module type configurations
# Each module type has: source_type_ids, type_names, display_name
MODULE_TYPES = {
    'gyro': {
        'name': 'Gyrostabilizer',
        'source_type_ids': {
            518,    # 'Basic' Gyrostabilizer
            519,    # Gyrostabilizer II
            520,    # Gyrostabilizer I
            5933,   # Counterbalanced Compact Gyrostabilizer
            13939,  # Domination Gyrostabilizer
            14536,  # Mizuro's Modified Gyrostabilizer
            14538,  # Hakim's Modified Gyrostabilizer
            14540,  # Gotan's Modified Gyrostabilizer
            14542,  # Tobias' Modified Gyrostabilizer
            15447,  # Shaqil's Modified Gyrostabilizer
            15806,  # Republic Fleet Gyrostabilizer
            21486,  # 'Kindred' Gyrostabilizer
            44112,  # Vadari's Custom Gyrostabilizer
        },
        'type_names': {
            518: "'Basic' Gyro",
            519: "Gyro II",
            520: "Gyro I",
            5933: "Compact Gyro",
            13939: "Domination Gyro",
            14536: "Mizuro's Gyro",
            14538: "Hakim's Gyro",
            14540: "Gotan's Gyro",
            14542: "Tobias' Gyro",
            15447: "Shaqil's Gyro",
            15806: "Republic Fleet",
            21486: "'Kindred' Gyro",
            44112: "Vadari's Gyro",
        },
    },
    'entropic': {
        'name': 'Entropic Radiation Sink',
        'source_type_ids': {
            47908,  # Entropic Radiation Sink I
            47911,  # Entropic Radiation Sink II
            48419,  # Veles Entropic Radiation Sink
            48421,  # Mystic Entropic Radiation Sink
        },
        'type_names': {
            47908: "Entropic I",
            47911: "Entropic II",
            48419: "Veles",
            48421: "Mystic",
        },
    },
    'heatsink': {
        'name': 'Heat Sink',
        'source_type_ids': {
            2363,   # Heat Sink I
            2364,   # Heat Sink II
            5849,   # Compact Heat Sink
            13943,  # Dark Blood Heat Sink
            13945,  # True Sansha Heat Sink
            14806,  # Ammatar Navy Heat Sink
            15810,  # Imperial Navy Heat Sink
            15812,  # Khanid Navy Heat Sink
        },
        'type_names': {
            2363: "Heat Sink I",
            2364: "Heat Sink II",
            5849: "Compact",
            13943: "Dark Blood",
            13945: "True Sansha",
            14806: "Ammatar Navy",
            15810: "Imperial Navy",
            15812: "Khanid Navy",
        },
    },
    'magstab': {
        'name': 'Magnetic Field Stabilizer',
        'source_type_ids': {
            9944,   # Magnetic Field Stabilizer I
            10190,  # Magnetic Field Stabilizer II
            5979,   # Compact Magnetic Field Stabilizer
            13947,  # Shadow Serpentis Mag Stab
            13949,  # Federation Navy Mag Stab
            15895,  # Federation Navy Magnetic Field Stabilizer
        },
        'type_names': {
            9944: "Mag Stab I",
            10190: "Mag Stab II",
            5979: "Compact",
            13947: "Shadow Serpentis",
            13949: "Fed Navy",
            15895: "Fed Navy",
        },
    },
    'bcs': {
        'name': 'Ballistic Control System',
        'source_type_ids': {
            22291,  # Ballistic Control System II
            22285,  # Ballistic Control System I
            22287,  # Compact BCS
            13935,  # Domination BCS
            13937,  # Republic Fleet BCS
            15681,  # Caldari Navy BCS
        },
        'type_names': {
            22291: "BCS II",
            22285: "BCS I",
            22287: "Compact",
            13935: "Domination",
            13937: "Republic Fleet",
            15681: "Caldari Navy",
        },
    },
}

# For backwards compatibility
GYROSTABILIZER_TYPE_IDS = MODULE_TYPES['gyro']['source_type_ids']
TYPE_NAMES = MODULE_TYPES['gyro']['type_names']


def download_contract_data(cache_dir: Path | None = None) -> Path:
    """Download the latest EVE Ref contract data archive."""
    if cache_dir is None:
        cache_dir = Path(tempfile.gettempdir()) / "everef_contracts"
    cache_dir.mkdir(parents=True, exist_ok=True)

    archive_path = cache_dir / "public-contracts-latest.v2.tar.bz2"

    # Check if we have a recent cache (less than 30 minutes old)
    if archive_path.exists():
        age_seconds = (datetime.now().timestamp() - archive_path.stat().st_mtime)
        if age_seconds < 1800:  # 30 minutes
            print(f"Using cached data (age: {age_seconds/60:.1f} minutes)")
            return archive_path

    print("Downloading latest contract data from EVE Ref...")
    response = requests.get(EVEREF_CONTRACTS_URL, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0

    with open(archive_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size:
                pct = (downloaded / total_size) * 100
                print(f"\rDownloading: {pct:.1f}%", end="", flush=True)

    print("\nDownload complete!")
    return archive_path


def extract_csv_from_archive(archive_path: Path, csv_name: str) -> pd.DataFrame:
    """Extract a specific CSV file from the tar.bz2 archive."""
    with tarfile.open(archive_path, 'r:bz2') as tar:
        for member in tar.getmembers():
            if member.name.endswith(csv_name):
                f = tar.extractfile(member)
                if f:
                    return pd.read_csv(f)
    raise FileNotFoundError(f"Could not find {csv_name} in archive")


def load_contract_data(archive_path: Path) -> ContractData:
    """Load all relevant CSV files from the archive."""
    print("Extracting contract data...")

    data: ContractData = {}  # type: ignore[typeddict-item]

    # Load contracts
    print("  Loading contracts...")
    data['contracts'] = extract_csv_from_archive(archive_path, 'contracts.csv')

    # Load contract items
    print("  Loading contract items...")
    data['items'] = extract_csv_from_archive(archive_path, 'contract_items.csv')

    # Load dynamic items (mutated modules)
    print("  Loading dynamic items...")
    data['dynamic_items'] = extract_csv_from_archive(archive_path, 'contract_dynamic_items.csv')

    # Load dogma attributes for dynamic items
    print("  Loading dogma attributes...")
    data['dogma_attributes'] = extract_csv_from_archive(archive_path, 'contract_dynamic_items_dogma_attributes.csv')

    return data


def find_mutated_modules(data: ContractData, module_type: str, source_type_id: int | None = None) -> pd.DataFrame:
    """Find all mutated modules of a given type in contracts with their attributes."""

    module_config = MODULE_TYPES.get(module_type)
    if not module_config:
        raise ValueError(f"Unknown module type: {module_type}. Available: {list(MODULE_TYPES.keys())}")

    contracts = data['contracts']
    dynamic_items = data['dynamic_items']
    dogma_attrs = data['dogma_attributes']

    # Filter for item exchange contracts
    available_contracts = contracts[
        contracts['type'] == 'item_exchange'
    ][['contract_id', 'price', 'issuer_id', 'date_issued', 'title']].copy()

    # Get dynamic items based on source type
    if source_type_id:
        # Filter for specific source type
        module_dynamic = dynamic_items[dynamic_items['source_type_id'] == source_type_id]
    else:
        # Filter for ALL source types of this module category
        module_dynamic = dynamic_items[dynamic_items['source_type_id'].isin(module_config['source_type_ids'])]

    if module_dynamic.empty:
        return pd.DataFrame()

    # Get dogma attributes for these items
    module_attrs = dogma_attrs[
        dogma_attrs['item_id'].isin(module_dynamic['item_id'])
    ]

    # Pivot to get damage modifier, ROF bonus, and CPU as columns
    damage_mod = module_attrs[module_attrs['attribute_id'] == ATTR_DAMAGE_MODIFIER][['item_id', 'value']]
    damage_mod = damage_mod.rename(columns={'value': 'damage_modifier'})

    rof_bonus = module_attrs[module_attrs['attribute_id'] == ATTR_ROF_BONUS][['item_id', 'value']]
    rof_bonus = rof_bonus.rename(columns={'value': 'rof_multiplier'})

    cpu = module_attrs[module_attrs['attribute_id'] == ATTR_CPU][['item_id', 'value']]
    cpu = cpu.rename(columns={'value': 'cpu'})

    # Merge attributes with dynamic items
    result = module_dynamic.merge(damage_mod, on='item_id', how='left')
    result = result.merge(rof_bonus, on='item_id', how='left')
    result = result.merge(cpu, on='item_id', how='left')

    # Dynamic items already have contract_id, so just merge with contracts for price
    result = result.merge(
        available_contracts,
        on='contract_id',
        how='inner'
    )

    # Convert ROF multiplier to bonus percentage
    # In EVE, speedMultiplier of 0.875 means 12.5% ROF bonus (1 - 0.875 = 0.125)
    if 'rof_multiplier' in result.columns:
        result['rof_bonus_pct'] = (1 - result['rof_multiplier']) * 100

    # Calculate DPS multiplier: damage_mod / speed_multiplier
    # This is the actual DPS contribution of the module
    if 'damage_modifier' in result.columns and 'rof_multiplier' in result.columns:
        result['dps_multiplier'] = result['damage_modifier'] / result['rof_multiplier']

    # Add source type name
    result['source_name'] = result['source_type_id'].map(module_config['type_names']).fillna('Unknown')

    return result


# Backwards compatibility alias
def find_mutated_gyrostabilizers(data: ContractData, source_type_id: int | None = None) -> pd.DataFrame:
    return find_mutated_modules(data, 'gyro', source_type_id)


def calculate_dps_multiplier(damage_mod: float, rof_bonus_pct: float) -> float:
    """Calculate the DPS multiplier from damage modifier and ROF bonus."""
    speed_multiplier = 1 - (rof_bonus_pct / 100)
    return damage_mod / speed_multiplier


def find_equivalent_or_worse(gyros: pd.DataFrame,
                              my_dps_mult: float,
                              my_cpu: float | None = None) -> pd.DataFrame:
    """
    Find gyrostabilizers with equivalent or worse DPS.

    For selling: you want to find items that are WORSE than yours
    to set a competitive price above them.

    Args:
        gyros: DataFrame of mutated gyros with attributes
        my_dps_mult: Your gyro's DPS multiplier
        my_cpu: Your gyro's CPU usage (e.g., 18.25) - lower is better, used for ties
    """
    # Filter for items with equal or worse DPS
    mask = gyros['dps_multiplier'] <= my_dps_mult

    # For items with similar DPS (within 0.5%), also consider CPU
    # CPU: lower is better, so "worse" means >= my_cpu
    if my_cpu is not None and 'cpu' in gyros.columns:
        # Items with significantly worse DPS, OR similar DPS but worse CPU
        similar_dps = (gyros['dps_multiplier'] >= my_dps_mult * 0.995)
        mask = mask & (~similar_dps | (gyros['cpu'] >= my_cpu))

    equivalent_or_worse = gyros[mask].copy()
    return equivalent_or_worse.sort_values('price')


def find_equivalent_or_better(gyros: pd.DataFrame,
                               my_dps_mult: float,
                               my_cpu: float | None = None) -> pd.DataFrame:
    """
    Find gyrostabilizers with equivalent or better DPS.

    For buying/competition: you want to find items that are BETTER than yours.
    """
    # Filter for items with equal or better DPS
    mask = gyros['dps_multiplier'] >= my_dps_mult

    # For items with similar DPS (within 0.5%), also consider CPU
    # CPU: lower is better, so "better" means <= my_cpu
    if my_cpu is not None and 'cpu' in gyros.columns:
        # Items with significantly better DPS, OR similar DPS but better CPU
        similar_dps = (gyros['dps_multiplier'] <= my_dps_mult * 1.005)
        mask = mask & (~similar_dps | (gyros['cpu'] <= my_cpu))

    equivalent_or_better = gyros[mask].copy()
    return equivalent_or_better.sort_values('price')


def format_isk(value: float) -> str:
    """Format ISK value with appropriate suffix."""
    if value >= 1_000_000_000:
        return f"{value/1_000_000_000:.2f}B"
    elif value >= 1_000_000:
        return f"{value/1_000_000:.2f}M"
    elif value >= 1_000:
        return f"{value/1_000:.2f}K"
    return f"{value:.0f}"


def display_results(results: pd.DataFrame, title: str, my_dps_mult: float | None = None) -> None:
    """Display results in a formatted table."""
    if results.empty:
        print(f"\n{title}")
        print("-" * 60)
        print("No matching contracts found.")
        return

    print(f"\n{title}")
    print("-" * 115)
    print(f"{'Price':>12} | {'DPS Mult':>8} | {'Damage':>7} | {'ROF':>7} | {'CPU':>6} | {'Source':<15} | Contract")
    print("-" * 115)

    for _, row in results.head(20).iterrows():
        price_str = format_isk(row['price'])
        dps = row.get('dps_multiplier', 0)
        dm = row.get('damage_modifier', 0)
        rof = row.get('rof_bonus_pct', 0)
        cpu_val = row.get('cpu', 0)
        source = row.get('source_name', 'Unknown')[:15]
        contract_id = row['contract_id']

        # Show DPS difference from yours if provided
        dps_diff = ""
        if my_dps_mult:
            diff_pct = ((dps / my_dps_mult) - 1) * 100
            dps_diff = f" ({diff_pct:+.1f}%)"

        print(f"{price_str:>12} | {dps:>8.4f}{dps_diff:>7} | {dm:>7.4f} | {rof:>6.2f}% | {cpu_val:>6.2f} | {source:<15} | {contract_id}")

    if len(results) > 20:
        print(f"... and {len(results) - 20} more")

    print("-" * 115)
    print(f"Total matching contracts: {len(results)}")

    if not results.empty:
        print(f"Lowest price: {format_isk(results['price'].min())}")
        print(f"Median price: {format_isk(results['price'].median())}")


def main() -> None:
    """Main entry point for the price finder CLI."""
    parser = argparse.ArgumentParser(
        description="Find lowest prices for equivalent mutated modules"
    )
    parser.add_argument(
        '--module', '-m',
        type=str,
        choices=list(MODULE_TYPES.keys()),
        default='gyro',
        help=f"Module type: {', '.join(MODULE_TYPES.keys())} (default: gyro)"
    )
    parser.add_argument(
        '--damage-mod', '-d',
        type=float,
        default=None,
        help='Your module damage modifier (e.g., 1.145)'
    )
    parser.add_argument(
        '--rof-bonus', '-r',
        type=float,
        default=None,
        help='Your module ROF bonus percentage (e.g., 12.49)'
    )
    parser.add_argument(
        '--cpu', '-c',
        type=float,
        default=None,
        help='Your module CPU usage (e.g., 18.25)'
    )
    parser.add_argument(
        '--show-better',
        action='store_true',
        help='Also show items with better stats (for comparison)'
    )
    parser.add_argument(
        '--cache-dir',
        type=Path,
        default=None,
        help='Directory to cache downloaded data'
    )

    args = parser.parse_args()

    module_config = MODULE_TYPES[args.module]
    module_name = module_config['name']

    # Default stats per module type if not provided
    defaults = {
        'gyro': {'damage': 1.145, 'rof': 12.49, 'cpu': 18.25},
        'entropic': {'damage': 1.12, 'rof': 8.0, 'cpu': 35.0},
        'heatsink': {'damage': 1.12, 'rof': 10.0, 'cpu': 30.0},
        'magstab': {'damage': 1.12, 'rof': 10.0, 'cpu': 30.0},
        'bcs': {'damage': 1.10, 'rof': 10.5, 'cpu': 40.0},
    }

    default = defaults.get(args.module, {'damage': 1.1, 'rof': 10.0, 'cpu': 30.0})
    damage_mod = args.damage_mod if args.damage_mod is not None else default['damage']
    rof_bonus = args.rof_bonus if args.rof_bonus is not None else default['rof']
    cpu = args.cpu if args.cpu is not None else default['cpu']

    # Calculate your DPS multiplier
    my_dps_mult = calculate_dps_multiplier(damage_mod, rof_bonus)

    print("=" * 60)
    print(f"EVE Online Mutated {module_name} Price Finder")
    print("=" * 60)
    print(f"\nYour stats:")
    print(f"  Damage Modifier: {damage_mod}")
    print(f"  ROF Bonus: {rof_bonus}%")
    print(f"  CPU: {cpu}")
    print(f"  DPS Multiplier: {my_dps_mult:.4f}")

    # Download and load data
    archive_path = download_contract_data(args.cache_dir)
    data = load_contract_data(archive_path)

    # Find mutated modules of the selected type
    modules = find_mutated_modules(data, args.module, source_type_id=None)

    if modules.empty:
        print(f"\nNo mutated {module_name}s found in current contracts!")
        return

    print(f"\nFound {len(modules)} mutated {module_name} contracts total")

    # Calculate DPS difference from yours for all items
    modules = modules.copy()
    modules['dps_diff_pct'] = ((modules['dps_multiplier'] / my_dps_mult) - 1) * 100

    # Sweetspot tier: -0.2% to 0% DPS (greedy competitive pricing)
    sweetspot = modules[(modules['dps_diff_pct'] >= -0.2) & (modules['dps_diff_pct'] < 0)].sort_values('price')

    # Items better than yours (competition)
    better = modules[modules['dps_diff_pct'] >= 0].sort_values('price')

    # Items worse than sweetspot (much worse than yours)
    worse = modules[modules['dps_diff_pct'] < -0.2].sort_values('price')

    # Display sweetspot tier
    print(f"\n{'='*60}")
    print("SWEETSPOT TIER (-0.2% to 0% DPS) - Your competition")
    print(f"{'='*60}")
    if not sweetspot.empty:
        print(f"{'Price':>12} | {'DPS':>8} | {'Diff':>6} | {'Dmg':>6} | {'ROF':>6} | {'CPU':>5} | Source")
        print("-" * 85)
        for _, row in sweetspot.iterrows():
            price_str = format_isk(row['price'])
            print(f"{price_str:>12} | {row['dps_multiplier']:.4f} | {row['dps_diff_pct']:+.2f}% | {row['damage_modifier']:.3f} | {row['rof_bonus_pct']:.2f}% | {row['cpu']:>5.1f} | {row['source_name']}")
        print("-" * 85)
        print(f"Count: {len(sweetspot)} | Lowest: {format_isk(sweetspot['price'].min())} | Median: {format_isk(sweetspot['price'].median())}")
    else:
        print("No items in sweetspot tier - you may be at the top!")

    # Display better items (brief)
    print(f"\n{'='*60}")
    print("BETTER THAN YOU (0%+ DPS)")
    print(f"{'='*60}")
    if not better.empty:
        for _, row in better.head(5).iterrows():
            price_str = format_isk(row['price'])
            print(f"{price_str:>12} | DPS {row['dps_multiplier']:.4f} ({row['dps_diff_pct']:+.2f}%) | CPU {row['cpu']:.1f} | {row['source_name']}")
        if len(better) > 5:
            print(f"... and {len(better) - 5} more")
        print(f"Cheapest better item: {format_isk(better['price'].min())}")
    else:
        print("No items better than yours - you're at the top!")

    # Pricing recommendation
    print(f"\n{'='*60}")
    print("PRICING RECOMMENDATION")
    print(f"{'='*60}")

    if not sweetspot.empty:
        sorted_prices = sweetspot['price'].sort_values().values
        competitive_price = sorted_prices[0]
        fair_price = sweetspot['price'].median()

        print(f"Competitive (quick sale): {format_isk(competitive_price)}")
        print(f"Fair value (median):      {format_isk(fair_price)}")

        # Find the gap between lowest and second lowest
        if len(sorted_prices) >= 2:
            lowest = sorted_prices[0]
            second_lowest = sorted_prices[1]
            gap_price = (lowest + second_lowest) / 2
            print(f"\nGap price strategy:")
            print(f"  Lowest:        {format_isk(lowest)}")
            print(f"  Second lowest: {format_isk(second_lowest)}")
            print(f"  --> YOUR PRICE: {format_isk(gap_price)}")
        else:
            print(f"\nOnly 1 item in tier - price at: {format_isk(competitive_price * 0.95)}")

    elif not better.empty:
        # No sweetspot items, price just below the cheapest better item
        print(f"Competitive: {format_isk(better['price'].min() * 0.95)}")
        print("(No items in your tier - pricing below competition)")
    else:
        print("You're at the top! Price as high as you want.")


if __name__ == '__main__':
    main()
