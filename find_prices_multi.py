#!/usr/bin/env python3
"""
EVE Online Mutated Module Price Finder - Multi-Module Version

Supports multiple module types for price comparison:
- Gyrostabilizers, Heat Sinks, Magnetic Field Stabilizers
- Ballistic Control Systems, Entropic Radiation Sinks
- Stasis Webifiers, Warp Disruptors, Warp Scramblers
- Shield Extenders, Armor Plates, Cap Batteries
- And more...

Uses EVE Ref public contract data which includes mutated item dogma attributes.
"""

import tarfile
import tempfile
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Callable, Optional

import requests
import pandas as pd

# EVE Ref data URL
EVEREF_CONTRACTS_URL = "https://data.everef.net/public-contracts/public-contracts-latest.v2.tar.bz2"
FUZZWORK_SDE = "https://www.fuzzwork.co.uk/dump/latest"

# Cache directory
CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

# Dogma Attribute IDs (verified from actual contract data)
ATTR_DAMAGE_MODIFIER = 64       # Damage Modifier (damage multiplier)
ATTR_ROF_MULTIPLIER = 204       # Rate of Fire Bonus (speed multiplier, lower = faster)
ATTR_CPU = 50                   # CPU usage
ATTR_CAPACITOR_BONUS = 67       # Capacitor Bonus (flat HP bonus)
ATTR_SHIELD_HP_BONUS = 72       # Shield Hitpoint Bonus
ATTR_ARMOR_HP_BONUS = 1159      # Armor Hitpoint Bonus
ATTR_MASS_ADDITION = 796        # Mass Addition (for armor plates)
ATTR_SIGNATURE_RADIUS_MOD = 983 # Signature Radius Modifier (for shield extenders)
ATTR_SPEED_FACTOR = 20          # Maximum Velocity Bonus (for webifiers, negative %)
ATTR_OPTIMAL_RANGE = 54         # Optimal Range (for webs, points, scrams)
ATTR_ACTIVATION_COST = 6        # Activation Cost (capacitor need)
ATTR_NEUT_RESISTANCE = 2267     # Capacitor Warfare Resistance Bonus (for cap batteries)
ATTR_DDA_DAMAGE = 1255          # Drone Damage Amplifier damage bonus


@dataclass
class ModuleConfig:
    """Configuration for a module type."""
    name: str
    short_name: str
    source_type_ids: set
    # Attributes: (attr_id, display_name, higher_is_better)
    primary_attr: tuple  # Main attribute for comparison
    secondary_attr: tuple  # Secondary attribute
    tertiary_attr: tuple  # Usually CPU
    # Function to calculate overall "score" for comparison
    calc_score: Callable


def calc_dps_score(row: pd.Series) -> float:
    """Calculate DPS score for damage modules.
    DPS contribution = damage_modifier / rof_multiplier
    Higher damage mod and lower ROF multiplier = more DPS.
    """
    if pd.isna(row.get('primary')) or pd.isna(row.get('secondary')):
        return 0
    # primary = damage modifier (e.g., 1.12)
    # secondary = ROF multiplier (e.g., 0.88 means 12% faster)
    return row['primary'] / row['secondary']


def calc_hp_score(row: pd.Series) -> float:
    """Calculate HP score for tank modules.
    Primary stat is the HP bonus - higher is better.
    """
    if pd.isna(row.get('primary')):
        return 0
    return row['primary']


def calc_web_score(row: pd.Series) -> float:
    """Calculate webifier score.
    Primary = velocity bonus (negative, e.g., -60 means 60% slow)
    Secondary = optimal range
    Score = |velocity bonus| * range factor
    """
    if pd.isna(row.get('primary')) or pd.isna(row.get('secondary')):
        return 0
    web_strength = abs(row['primary'])  # Convert -60 to 60
    range_km = row['secondary'] / 1000  # Convert to km
    return web_strength * range_km  # e.g., 60% * 14km = 840


def calc_range_score(row: pd.Series) -> float:
    """Calculate range score for points/scrams.
    Primary = optimal range - higher is better.
    """
    if pd.isna(row.get('primary')):
        return 0
    return row['primary'] / 1000  # Return in km for readability


def calc_cap_battery_score(row: pd.Series) -> float:
    """Calculate cap battery score.
    Primary = capacitor bonus (flat HP)
    Secondary = neut resistance (more negative = better)
    """
    if pd.isna(row.get('primary')):
        return 0
    cap_bonus = row['primary']
    neut_resist = abs(row.get('secondary', 0) or 0)  # e.g., -32 -> 32
    # Weight: cap bonus matters most, neut resist is secondary
    return cap_bonus + (neut_resist * 20)  # 32% resist adds ~640 to score


# Module type configurations (source_type_ids verified from contract data)
MODULE_CONFIGS = {
    # === DAMAGE MODULES (DPS = damage_mod / rof_multiplier) ===
    'gyro': ModuleConfig(
        name='Gyrostabilizer',
        short_name='Gyro',
        source_type_ids={518, 519, 520, 5933, 13939, 14536, 14538, 14540, 14542, 15447, 15806, 21486, 44112},
        primary_attr=(ATTR_DAMAGE_MODIFIER, 'Damage Mod', True),
        secondary_attr=(ATTR_ROF_MULTIPLIER, 'ROF Multi', False),
        tertiary_attr=(ATTR_CPU, 'CPU', False),
        calc_score=calc_dps_score,
    ),
    'heatsink': ModuleConfig(
        name='Heat Sink',
        short_name='HS',
        # Verified from contract data:
        # 2363: Heat Sink I, 2364: Heat Sink II, 5849: Compact
        # 13941: Dark Blood, 13943: True Sansha, 15810: Imperial Navy
        # 14804: Selynne's, 14806: Brynn's, 14808: Tuvan's, 14810: Ahremen's, 14812: Chelm's (Officers)
        source_type_ids={2363, 2364, 5849, 13941, 13943, 15810, 14804, 14806, 14808, 14810, 14812},
        primary_attr=(ATTR_DAMAGE_MODIFIER, 'Damage Mod', True),
        secondary_attr=(ATTR_ROF_MULTIPLIER, 'ROF Multi', False),
        tertiary_attr=(ATTR_CPU, 'CPU', False),
        calc_score=calc_dps_score,
    ),
    'magstab': ModuleConfig(
        name='Magnetic Field Stabilizer',
        short_name='Magstab',
        source_type_ids={9944, 9946, 9948, 10190, 14680, 14682, 14684, 14686, 15416, 15940, 22291, 44110},
        primary_attr=(ATTR_DAMAGE_MODIFIER, 'Damage Mod', True),
        secondary_attr=(ATTR_ROF_MULTIPLIER, 'ROF Multi', False),
        tertiary_attr=(ATTR_CPU, 'CPU', False),
        calc_score=calc_dps_score,
    ),
    'bcs': ModuleConfig(
        name='Ballistic Control System',
        short_name='BCS',
        source_type_ids={12269, 12275, 16457, 22291, 22303, 22305, 22307, 22309, 28563, 33844, 44114},
        primary_attr=(ATTR_DAMAGE_MODIFIER, 'Damage Mod', True),
        secondary_attr=(ATTR_ROF_MULTIPLIER, 'ROF Multi', False),
        tertiary_attr=(ATTR_CPU, 'CPU', False),
        calc_score=calc_dps_score,
    ),
    'entropic': ModuleConfig(
        name='Entropic Radiation Sink',
        short_name='ERS',
        source_type_ids={47908, 47909, 47910, 47911},
        primary_attr=(ATTR_DAMAGE_MODIFIER, 'Damage Mod', True),
        secondary_attr=(ATTR_ROF_MULTIPLIER, 'ROF Multi', False),
        tertiary_attr=(ATTR_CPU, 'CPU', False),
        calc_score=calc_dps_score,
    ),
    'dda': ModuleConfig(
        name='Drone Damage Amplifier',
        short_name='DDA',
        # DDA source types: T1, T2, Compact, Dread Guristas, Sentient, Fed Navy
        # 33836: DDA I, 4405: DDA II, 33837: Packrat Compact
        # 33848: Dread Guristas, 33846: Sentient, 33844: Fed Navy
        source_type_ids={33836, 4405, 33837, 33848, 33846, 33844},
        primary_attr=(ATTR_DDA_DAMAGE, 'Dmg Bonus', True),  # e.g., 1.25 = 25% bonus
        secondary_attr=(ATTR_CPU, 'CPU', False),  # DDAs only have damage + CPU
        tertiary_attr=(ATTR_CPU, 'CPU', False),  # Duplicate, but needed for structure
        calc_score=calc_hp_score,  # Just use primary value as score (higher damage = better)
    ),

    # === EWAR MODULES ===
    'web': ModuleConfig(
        name='Stasis Webifier',
        short_name='Web',
        source_type_ids={526, 527, 4025, 4027, 14262, 14264, 14266, 14268, 17500, 17559, 14260},
        primary_attr=(ATTR_SPEED_FACTOR, 'Web %', False),  # e.g., -60 = 60% slow
        secondary_attr=(ATTR_OPTIMAL_RANGE, 'Range', True),
        tertiary_attr=(ATTR_CPU, 'CPU', False),
        calc_score=calc_web_score,
    ),
    'point': ModuleConfig(
        name='Warp Disruptor',
        short_name='Point',
        source_type_ids={447, 448, 3242, 5399, 14244, 14246, 14248, 14250, 17559, 28516},
        primary_attr=(ATTR_OPTIMAL_RANGE, 'Range', True),  # Range is the key stat
        secondary_attr=(ATTR_ACTIVATION_COST, 'Cap Use', False),
        tertiary_attr=(ATTR_CPU, 'CPU', False),
        calc_score=calc_range_score,
    ),
    'scram': ModuleConfig(
        name='Warp Scrambler',
        short_name='Scram',
        source_type_ids={447, 3242, 5399, 14236, 14238, 14240, 14242, 17559, 28514},
        primary_attr=(ATTR_OPTIMAL_RANGE, 'Range', True),
        secondary_attr=(ATTR_ACTIVATION_COST, 'Cap Use', False),
        tertiary_attr=(ATTR_CPU, 'CPU', False),
        calc_score=calc_range_score,
    ),

    # === TANK MODULES (HP is the key stat) ===
    'lse': ModuleConfig(
        name='Large Shield Extender',
        short_name='LSE',
        source_type_ids={3841, 8529, 31930, 31932},  # LSE II, Compact, CN, RF
        primary_attr=(ATTR_SHIELD_HP_BONUS, 'Shield HP', True),
        secondary_attr=(ATTR_SIGNATURE_RADIUS_MOD, 'Sig Bloom', False),
        tertiary_attr=(ATTR_CPU, 'CPU', False),
        calc_score=calc_hp_score,
    ),
    'plate': ModuleConfig(
        name='1600mm Armor Plate',
        short_name='Plate',
        source_type_ids={20349, 20351, 20353, 22299, 22301, 23795, 23797},  # Various 1600mm plates
        primary_attr=(ATTR_ARMOR_HP_BONUS, 'Armor HP', True),
        secondary_attr=(ATTR_MASS_ADDITION, 'Mass Add', False),
        tertiary_attr=(ATTR_CPU, 'CPU', False),
        calc_score=calc_hp_score,
    ),

    # === CAPACITOR MODULES ===
    'capbat': ModuleConfig(
        name='Large Cap Battery',
        short_name='CapBat',
        source_type_ids={3504, 41218, 41220, 23805},  # Large Cap Bat II, RF, Thukker, Thurifer
        primary_attr=(ATTR_CAPACITOR_BONUS, 'Cap Bonus', True),
        secondary_attr=(ATTR_NEUT_RESISTANCE, 'Neut Resist', False),  # More negative = better
        tertiary_attr=(ATTR_CPU, 'CPU', False),
        calc_score=calc_cap_battery_score,
    ),
}


def download_contract_data(cache_dir: Path = None) -> Path:
    """Download the latest EVE Ref contract data archive."""
    if cache_dir is None:
        cache_dir = CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    archive_path = cache_dir / "public-contracts-latest.v2.tar.bz2"

    if archive_path.exists():
        age_seconds = (datetime.now().timestamp() - archive_path.stat().st_mtime)
        if age_seconds < 1800:
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


def load_contract_data(archive_path: Path) -> dict:
    """Load all relevant CSV files from the archive."""
    print("Extracting contract data...")

    data = {}
    print("  Loading contracts...")
    data['contracts'] = extract_csv_from_archive(archive_path, 'contracts.csv')
    print("  Loading dynamic items...")
    data['dynamic_items'] = extract_csv_from_archive(archive_path, 'contract_dynamic_items.csv')
    print("  Loading dogma attributes...")
    data['dogma_attributes'] = extract_csv_from_archive(archive_path, 'contract_dynamic_items_dogma_attributes.csv')

    return data


def load_type_names() -> dict:
    """Load type names from SDE."""
    cache_path = CACHE_DIR / "invTypes.csv"

    if not cache_path.exists():
        print("  Downloading type names...")
        response = requests.get(f"{FUZZWORK_SDE}/invTypes.csv", timeout=60)
        response.raise_for_status()
        cache_path.write_bytes(response.content)

    types_df = pd.read_csv(cache_path)
    return dict(zip(types_df['typeID'], types_df['typeName']))


def find_mutated_modules(data: dict, config: ModuleConfig) -> pd.DataFrame:
    """Find all mutated modules of a specific type with their attributes."""

    contracts = data['contracts']
    dynamic_items = data['dynamic_items']
    dogma_attrs = data['dogma_attributes']

    # Filter for item exchange contracts
    available_contracts = contracts[
        contracts['type'] == 'item_exchange'
    ][['contract_id', 'price', 'issuer_id', 'date_issued', 'title']].copy()

    # Filter for modules of this type
    modules = dynamic_items[dynamic_items['source_type_id'].isin(config.source_type_ids)]

    if modules.empty:
        return pd.DataFrame()

    # Get dogma attributes for these items
    module_attrs = dogma_attrs[dogma_attrs['item_id'].isin(modules['item_id'])]

    # Extract primary attribute
    primary = module_attrs[module_attrs['attribute_id'] == config.primary_attr[0]][['item_id', 'value']]
    primary = primary.rename(columns={'value': 'primary'})

    # Extract secondary attribute
    secondary = module_attrs[module_attrs['attribute_id'] == config.secondary_attr[0]][['item_id', 'value']]
    secondary = secondary.rename(columns={'value': 'secondary'})

    # Extract tertiary (usually CPU)
    tertiary = module_attrs[module_attrs['attribute_id'] == config.tertiary_attr[0]][['item_id', 'value']]
    tertiary = tertiary.rename(columns={'value': 'tertiary'})

    # Merge attributes
    result = modules.merge(primary, on='item_id', how='left')
    result = result.merge(secondary, on='item_id', how='left')
    result = result.merge(tertiary, on='item_id', how='left')

    # Merge with contracts
    result = result.merge(available_contracts, on='contract_id', how='inner')

    # Calculate score
    result['score'] = result.apply(config.calc_score, axis=1)

    return result


def format_isk(value: float) -> str:
    """Format ISK value with appropriate suffix."""
    if value >= 1_000_000_000:
        return f"{value/1_000_000_000:.2f}B"
    elif value >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    elif value >= 1_000:
        return f"{value/1_000:.1f}K"
    return f"{value:.0f}"


def main():
    parser = argparse.ArgumentParser(
        description="Find lowest prices for equivalent mutated modules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Module types:
  gyro      - Gyrostabilizer (projectile damage)
  heatsink  - Heat Sink (laser damage)
  magstab   - Magnetic Field Stabilizer (hybrid damage)
  bcs       - Ballistic Control System (missile damage)
  entropic  - Entropic Radiation Sink (vorton damage)
  dda       - Drone Damage Amplifier (drone damage)
  web       - Stasis Webifier
  point     - Warp Disruptor
  scram     - Warp Scrambler
  lse       - Large Shield Extender
  plate     - 1600mm Armor Plate
  capbat    - Large Cap Battery

Examples:
  python find_prices_multi.py gyro --primary 1.145 --secondary 0.8751
  python find_prices_multi.py dda --primary 1.25 --secondary 20
  python find_prices_multi.py web --primary -60 --secondary 14000
  python find_prices_multi.py lse --primary 2250 --secondary 12
        """
    )
    parser.add_argument(
        'module_type',
        choices=list(MODULE_CONFIGS.keys()),
        help='Type of module to search for'
    )
    parser.add_argument(
        '--primary', '-p',
        type=float,
        required=True,
        help='Your module\'s primary attribute value'
    )
    parser.add_argument(
        '--secondary', '-s',
        type=float,
        default=None,
        help='Your module\'s secondary attribute value (for ROF, use the %% shown in EVE, e.g., 12.5)'
    )
    parser.add_argument(
        '--rof-percent',
        action='store_true',
        help='Treat secondary as ROF percentage (converts 12.5 to 0.875 multiplier)'
    )
    parser.add_argument(
        '--tertiary', '-t',
        type=float,
        default=None,
        help='Your module\'s tertiary attribute value (usually CPU)'
    )
    parser.add_argument(
        '--tolerance',
        type=float,
        default=0.5,
        help='Percentage tolerance for "similar" items (default: 0.5%%)'
    )

    args = parser.parse_args()

    config = MODULE_CONFIGS[args.module_type]

    print("=" * 70)
    print(f"EVE Online Mutated {config.name} Price Finder")
    print("=" * 70)

    # Calculate your score
    my_primary = args.primary
    my_secondary = args.secondary
    my_tertiary = args.tertiary

    # Convert ROF percentage to multiplier for damage modules
    # EVE shows "12.5%" but internally it's stored as 0.875 multiplier
    # Auto-convert if secondary looks like a percentage (> 1) for damage modules
    damage_modules = {'gyro', 'heatsink', 'magstab', 'bcs', 'entropic'}
    if my_secondary is not None and args.module_type in damage_modules:
        if my_secondary > 1 or args.rof_percent:
            # User gave percentage like 12.5, convert to multiplier 0.875
            my_secondary = 1 - (my_secondary / 100)
            print(f"  (Converted ROF {args.secondary}% to multiplier {my_secondary:.4f})")

    # Create a fake row to calculate score
    my_row = pd.Series({'primary': my_primary, 'secondary': my_secondary, 'tertiary': my_tertiary})
    my_score = config.calc_score(my_row)

    print(f"\nYour stats:")
    print(f"  {config.primary_attr[1]}: {my_primary}")
    if my_secondary is not None:
        print(f"  {config.secondary_attr[1]}: {my_secondary}")
    if my_tertiary is not None:
        print(f"  {config.tertiary_attr[1]}: {my_tertiary}")
    print(f"  Score: {my_score:.4f}")

    # Download and load data
    archive_path = download_contract_data()
    data = load_contract_data(archive_path)

    # Load type names for display
    type_names = load_type_names()

    # Find mutated modules
    modules = find_mutated_modules(data, config)

    if modules.empty:
        print(f"\nNo mutated {config.name}s found in current contracts!")
        return

    # Add source names
    modules['source_name'] = modules['source_type_id'].map(type_names).fillna('Unknown')
    # Shorten names for display
    modules['source_short'] = modules['source_name'].str.replace('Domination ', 'Dom ').str.replace('Republic Fleet ', 'RF ').str.replace('Federation Navy ', 'FN ').str.replace('Imperial Navy ', 'IN ').str.replace('Caldari Navy ', 'CN ').str.replace('Thukker ', 'Thuk ')

    print(f"\nFound {len(modules)} mutated {config.name} contracts")

    # Calculate score difference
    modules = modules.copy()
    modules['score_diff_pct'] = ((modules['score'] / my_score) - 1) * 100

    # Categorize
    tolerance = args.tolerance
    sweetspot = modules[(modules['score_diff_pct'] >= -tolerance) & (modules['score_diff_pct'] < 0)].sort_values('price')
    better = modules[modules['score_diff_pct'] >= 0].sort_values('price')
    worse = modules[modules['score_diff_pct'] < -tolerance].sort_values('price')

    # Display sweetspot
    print(f"\n{'='*70}")
    print(f"SWEETSPOT TIER (-{tolerance}% to 0%) - Your competition")
    print(f"{'='*70}")

    if not sweetspot.empty:
        print(f"{'Price':>10} | {'Score':>8} | {'Diff':>7} | {config.primary_attr[1][:8]:>8} | {config.secondary_attr[1][:8]:>8} | Source")
        print("-" * 75)
        for _, row in sweetspot.head(15).iterrows():
            print(f"{format_isk(row['price']):>10} | {row['score']:.4f} | {row['score_diff_pct']:+.2f}% | {row['primary']:>8.2f} | {row['secondary']:>8.2f} | {row['source_short'][:20]}")
        print("-" * 75)
        print(f"Count: {len(sweetspot)} | Lowest: {format_isk(sweetspot['price'].min())} | Median: {format_isk(sweetspot['price'].median())}")
    else:
        print("No items in sweetspot tier")

    # Display better items
    print(f"\n{'='*70}")
    print("BETTER THAN YOURS")
    print(f"{'='*70}")

    if not better.empty:
        for _, row in better.head(5).iterrows():
            print(f"{format_isk(row['price']):>10} | Score {row['score']:.4f} ({row['score_diff_pct']:+.2f}%) | {row['source_short'][:25]}")
        if len(better) > 5:
            print(f"... and {len(better) - 5} more")
        print(f"Cheapest better: {format_isk(better['price'].min())}")
    else:
        print("No items better than yours - you're at the top!")

    # Pricing recommendation
    print(f"\n{'='*70}")
    print("PRICING RECOMMENDATION")
    print(f"{'='*70}")

    if not sweetspot.empty:
        sorted_prices = sweetspot['price'].sort_values().values
        print(f"Competitive (quick): {format_isk(sorted_prices[0])}")
        print(f"Fair (median):       {format_isk(sweetspot['price'].median())}")

        if len(sorted_prices) >= 2:
            gap = (sorted_prices[0] + sorted_prices[1]) / 2
            print(f"\nGap strategy:")
            print(f"  Lowest:  {format_isk(sorted_prices[0])}")
            print(f"  Second:  {format_isk(sorted_prices[1])}")
            print(f"  --> YOUR PRICE: {format_isk(gap)}")
    elif not better.empty:
        print(f"Price below competition: {format_isk(better['price'].min() * 0.95)}")
    else:
        print("You're at the top! Price as high as you want.")


if __name__ == '__main__':
    main()
