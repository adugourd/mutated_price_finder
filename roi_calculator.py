#!/usr/bin/env python3
"""
EVE Online Mutated Module ROI Calculator

Calculates expected ROI for rolling mutated modules based on:
1. Base item cost (Jita sell orders)
2. Mutaplasmid cost (Jita sell orders)
3. Success rate based on probability of rolling "good enough" stats
4. Expected sale price based on contract distribution

Uses Monte Carlo simulation to calculate success probability where:
- Primary stats need to be >= median threshold (e.g., DPS for damage mods)
- Secondary stats must NOT be catastrophic (not in worst 10% of rolls)
"""

import tarfile
import argparse
from pathlib import Path
from datetime import datetime

import requests
import pandas as pd
import numpy as np
from scipy import stats

# Import configuration from YAML files
from src.config.loader import (
    load_all_targets,
    load_constants,
    load_attributes,
    MutaplasmidRange,
    RollTarget,
)

# Load constants from config
_constants = load_constants()
_attributes = load_attributes()

# Cache directory
CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

# Fuzzwork market API (Jita = The Forge region 10000002)
FUZZWORK_MARKET_URL = _constants['api']['fuzzwork_market_url']
JITA_REGION = _constants['api']['jita_region_id']

# EVE Ref contracts
EVEREF_CONTRACTS_URL = _constants['api']['everef_contracts_url']

# Number of Monte Carlo samples
NUM_SAMPLES = _constants['simulation']['num_samples']

# Attribute IDs (loaded from config for backward compatibility)
ATTR_DAMAGE = _attributes['damage']
ATTR_ROF = _attributes['rof']
ATTR_CPU = _attributes['cpu']
ATTR_VELOCITY_BONUS = _attributes['velocity_bonus']
ATTR_CAP = _attributes['cap']
ATTR_RANGE = _attributes['range']
ATTR_SHIELD_BOOST = _attributes['shield_boost']
ATTR_DURATION = _attributes['duration']
ATTR_ARMOR_HP = _attributes['armor_hp']
ATTR_MASS = _attributes['mass']
ATTR_POWER = _attributes['power']
ATTR_DDA_DAMAGE = _attributes['dda_damage']
ATTR_SHIELD_CAP = _attributes['shield_cap']
ATTR_CAP_CAPACITY = _attributes['cap_capacity']
ATTR_MISSILE_DAMAGE = _attributes['missile_damage']


# Load roll targets from YAML configuration files
# Configuration files are in config/targets/*.yaml
ROLL_TARGETS = load_all_targets()


# Legacy compatibility: Also provide access to individual targets
def _get_roll_target(key: str) -> RollTarget:
    """Get a specific roll target by key (for backward compatibility)."""
    return ROLL_TARGETS[key]


# NOTE: The following ROLL_TARGETS dict has been moved to config/targets/*.yaml
# To add new targets, edit the YAML files instead of this code.
# See config/targets/ for examples.



def get_jita_prices(type_ids: list) -> dict:
    """Get Jita sell prices for multiple type IDs."""
    type_str = ','.join(str(t) for t in type_ids)
    url = f"{FUZZWORK_MARKET_URL}?region={JITA_REGION}&types={type_str}"

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()

    prices = {}
    for type_id in type_ids:
        str_id = str(type_id)
        if str_id in data:
            sell_data = data[str_id].get('sell', {})
            prices[type_id] = {
                'min': float(sell_data.get('min', 0)),
                'median': float(sell_data.get('median', 0)),
                'percentile': float(sell_data.get('percentile', 0)),
            }
        else:
            prices[type_id] = {'min': 0, 'median': 0, 'percentile': 0}

    return prices


def download_contract_data() -> Path:
    """Download contract data if not cached."""
    archive_path = CACHE_DIR / "public-contracts-latest.v2.tar.bz2"

    if archive_path.exists():
        age_seconds = (datetime.now().timestamp() - archive_path.stat().st_mtime)
        if age_seconds < 1800:
            return archive_path

    print("Downloading contract data...")
    response = requests.get(EVEREF_CONTRACTS_URL, stream=True, timeout=120)
    response.raise_for_status()

    with open(archive_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return archive_path


def extract_csv_from_archive(archive_path: Path, csv_name: str) -> pd.DataFrame:
    """Extract a specific CSV file from the archive."""
    with tarfile.open(archive_path, 'r:bz2') as tar:
        for member in tar.getmembers():
            if member.name.endswith(csv_name):
                f = tar.extractfile(member)
                if f:
                    return pd.read_csv(f)
    raise FileNotFoundError(f"Could not find {csv_name} in archive")


def find_realistic_sale_price_stat_based(
    stats: np.ndarray,
    prices: np.ndarray,
    base_stat: float,
    max_stat: float,
) -> dict:
    """
    Find realistic expected sale price using stat-based methodology.

    1. Filter to sellable items (stat > base_stat)
    2. Remove price outliers using IQR method FIRST
    3. Fit linear regression constrained through (base_stat, worst_sellable_price)
    4. Expected price = fitted value at midpoint of [base_stat, max_stat] range

    Returns dict with analysis details
    """
    if len(stats) < 3:
        return {
            'n_total': len(stats),
            'n_sellable': 0,
            'n_outliers': 0,
            'n_used': 0,
            'expected_price': 0,
            'midpoint_stat': (base_stat + max_stat) / 2,
            'slope': 0,
            'anchor_price': 0,
            'coverage_pct': 0,
            'method': 'insufficient_data',
        }

    # Step 1: Filter to sellable items (stat > base_stat)
    sellable_mask = stats > base_stat
    sellable_stats = stats[sellable_mask]
    sellable_prices = prices[sellable_mask]
    n_sellable = len(sellable_stats)

    if n_sellable < 3:
        # Calculate coverage even for limited data
        full_range = max_stat - base_stat
        if n_sellable > 0 and full_range > 0:
            data_min = np.min(sellable_stats)
            data_max = np.max(sellable_stats)
            min_pct = (data_min - base_stat) / full_range * 100
            max_pct = (data_max - base_stat) / full_range * 100
            coverage = max_pct - min_pct
        else:
            coverage = 0
        return {
            'n_total': len(stats),
            'n_sellable': n_sellable,
            'n_outliers': 0,
            'n_used': n_sellable,
            'expected_price': np.mean(sellable_prices) if n_sellable > 0 else 0,
            'midpoint_stat': (base_stat + max_stat) / 2,
            'slope': 0,
            'anchor_price': np.min(sellable_prices) if n_sellable > 0 else 0,
            'coverage_pct': coverage,
            'method': 'simple_mean',
        }

    # Step 2: Remove price outliers using IQR method FIRST
    q1 = np.percentile(sellable_prices, 25)
    q3 = np.percentile(sellable_prices, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    inlier_mask = (sellable_prices >= lower_bound) & (sellable_prices <= upper_bound)
    clean_stats = sellable_stats[inlier_mask]
    clean_prices = sellable_prices[inlier_mask]
    n_outliers = n_sellable - len(clean_stats)

    if len(clean_stats) < 2:
        # Fall back to simple mean of inliers
        full_range = max_stat - base_stat
        if len(clean_stats) > 0 and full_range > 0:
            data_min = np.min(clean_stats)
            data_max = np.max(clean_stats)
            min_pct = (data_min - base_stat) / full_range * 100
            max_pct = (data_max - base_stat) / full_range * 100
            coverage = max_pct - min_pct
        else:
            coverage = 0
        return {
            'n_total': len(stats),
            'n_sellable': n_sellable,
            'n_outliers': n_outliers,
            'n_used': len(clean_stats),
            'expected_price': np.mean(clean_prices) if len(clean_prices) > 0 else 0,
            'midpoint_stat': (base_stat + max_stat) / 2,
            'slope': 0,
            'anchor_price': np.min(clean_prices) if len(clean_prices) > 0 else 0,
            'coverage_pct': coverage,
            'method': 'mean_after_outlier_removal',
        }

    # Step 3: Calculate stat range coverage confidence
    # Coverage = how much of the sellable range [base_stat, max_stat] is covered by data
    full_range = max_stat - base_stat
    data_min = np.min(clean_stats)
    data_max = np.max(clean_stats)

    # Calculate percentile positions within the full range
    min_percentile = (data_min - base_stat) / full_range * 100 if full_range > 0 else 0
    max_percentile = (data_max - base_stat) / full_range * 100 if full_range > 0 else 0
    coverage_pct = max_percentile - min_percentile

    # Step 4: Fit linear regression constrained through (base_stat, anchor_price)
    # Anchor price = price of worst sellable item (lowest stat among clean data)
    worst_idx = np.argmin(clean_stats)
    anchor_stat = clean_stats[worst_idx]
    anchor_price = clean_prices[worst_idx]

    # Transform data to pass through anchor point: y - anchor_price = slope * (x - anchor_stat)
    # Fit no-intercept regression on transformed data
    x_shifted = clean_stats - anchor_stat
    y_shifted = clean_prices - anchor_price

    # Least squares: slope = sum(x*y) / sum(x^2)
    if np.sum(x_shifted ** 2) > 0:
        slope = np.sum(x_shifted * y_shifted) / np.sum(x_shifted ** 2)
    else:
        slope = 0

    # Step 5: Calculate expected price at midpoint of [base_stat, max_stat] range
    midpoint_stat = (base_stat + max_stat) / 2
    # Price at midpoint using the constrained line through anchor
    expected_price = anchor_price + slope * (midpoint_stat - anchor_stat)

    # Sanity check: expected price should be positive
    if expected_price < 0:
        expected_price = np.mean(clean_prices)

    return {
        'n_total': len(stats),
        'n_sellable': n_sellable,
        'n_outliers': n_outliers,
        'n_used': len(clean_stats),
        'expected_price': expected_price,
        'midpoint_stat': midpoint_stat,
        'slope': slope,
        'anchor_stat': anchor_stat,
        'anchor_price': anchor_price,
        'base_stat': base_stat,
        'max_stat': max_stat,
        'data_min_stat': data_min,
        'data_max_stat': data_max,
        'coverage_pct': coverage_pct,
        'method': 'constrained_regression',
    }


def find_realistic_sale_price(prices: np.ndarray, roll_cost: float) -> dict:
    """
    Find realistic expected sale price using conservative methodology.
    (Legacy method - kept for backwards compatibility)

    1. Filter to prices above material cost (sellable rolls only)
    2. Cap at 2x median to remove outliers
    3. Return mean of filtered prices

    Returns dict with analysis details
    """
    if len(prices) < 3:
        # Not enough data
        median = np.median(prices) if len(prices) > 0 else 0
        return {
            'sellable_prices': prices,
            'median': median,
            'capped_mean': median,
            'n_total': len(prices),
            'n_below_cost': 0,
            'n_outliers': 0,
            'n_used': len(prices),
            'outlier_cap': median * 2,
        }

    # Step 1: Filter to prices above material cost (sellable)
    sellable = prices[prices >= roll_cost]
    n_below_cost = len(prices) - len(sellable)

    if len(sellable) == 0:
        return {
            'sellable_prices': np.array([]),
            'median': 0,
            'capped_mean': 0,
            'n_total': len(prices),
            'n_below_cost': n_below_cost,
            'n_outliers': 0,
            'n_used': 0,
            'outlier_cap': 0,
        }

    # Step 2: Cap at 2x median to remove outliers
    median = np.median(sellable)
    outlier_cap = median * 2
    capped = sellable[sellable <= outlier_cap]
    n_outliers = len(sellable) - len(capped)

    # Step 3: Calculate mean
    capped_mean = np.mean(capped) if len(capped) > 0 else median

    return {
        'sellable_prices': sellable,
        'median': median,
        'capped_mean': capped_mean,
        'n_total': len(prices),
        'n_below_cost': n_below_cost,
        'n_outliers': n_outliers,
        'n_used': len(capped),
        'outlier_cap': outlier_cap,
    }


def analyze_contract_distribution(archive_path: Path, source_type_id: int, muta_type_id: int, roll_cost: float) -> dict:
    """Analyze contract price distribution for a specific item+muta combo (legacy method)."""

    contracts = extract_csv_from_archive(archive_path, 'contracts.csv')
    dynamic_items = extract_csv_from_archive(archive_path, 'contract_dynamic_items.csv')

    # Filter for item exchange contracts
    item_contracts = contracts[contracts['type'] == 'item_exchange'][['contract_id', 'price']].copy()

    # Filter for this specific source+muta combo
    matching_items = dynamic_items[
        (dynamic_items['source_type_id'] == source_type_id) &
        (dynamic_items['mutator_type_id'] == muta_type_id)
    ]

    # Only keep contracts that have exactly 1 dynamic item TOTAL
    # (discard any bundles - contracts with multiple items of any type)
    total_items_per_contract = dynamic_items.groupby('contract_id').size()
    single_item_contracts = total_items_per_contract[total_items_per_contract == 1].index
    matching_items = matching_items[matching_items['contract_id'].isin(single_item_contracts)]

    # Get prices for matching contracts
    prices_df = matching_items.merge(item_contracts, on='contract_id', how='inner')

    if prices_df.empty:
        return {
            'count': 0,
            'prices': np.array([]),
            'price_analysis': {
                'median': 0,
                'capped_mean': 0,
                'n_total': 0,
                'n_below_cost': 0,
                'n_outliers': 0,
                'n_used': 0,
                'outlier_cap': 0,
            },
        }

    prices = prices_df['price'].values

    # Analyze prices with conservative methodology
    price_analysis = find_realistic_sale_price(prices, roll_cost)

    return {
        'count': len(prices),
        'prices': prices,
        'price_analysis': price_analysis,
    }


def analyze_contracts_stat_based(
    archive_path: Path,
    muta_type_id: int,
    primary_attr_id: int,
    base_stat: float,
    max_stat: float,
) -> dict:
    """
    Analyze contract prices using stat-based filtering.

    Looks at ALL mutated items with the given mutaplasmid (any source type),
    filters by actual stat values, and uses regression-based pricing.

    Args:
        archive_path: Path to contract archive
        muta_type_id: Mutaplasmid type ID
        primary_attr_id: Primary attribute ID to filter on (e.g., 1255 for DDA damage)
        base_stat: Base item stat value (sellable threshold)
        max_stat: Maximum possible stat value (base * max_roll_mult)

    Returns:
        Dict with analysis results
    """
    contracts = extract_csv_from_archive(archive_path, 'contracts.csv')
    dynamic_items = extract_csv_from_archive(archive_path, 'contract_dynamic_items.csv')
    dogma_attrs = extract_csv_from_archive(archive_path, 'contract_dynamic_items_dogma_attributes.csv')

    # Filter for item exchange contracts
    item_contracts = contracts[contracts['type'] == 'item_exchange'][['contract_id', 'price']].copy()

    # Get ALL items with this mutaplasmid (any source type)
    matching_items = dynamic_items[dynamic_items['mutator_type_id'] == muta_type_id].copy()

    # Only keep contracts with exactly 1 dynamic item (no bundles)
    total_items_per_contract = dynamic_items.groupby('contract_id').size()
    single_item_contracts = total_items_per_contract[total_items_per_contract == 1].index
    matching_items = matching_items[matching_items['contract_id'].isin(single_item_contracts)]

    if matching_items.empty:
        return {
            'count': 0,
            'stats': np.array([]),
            'prices': np.array([]),
            'stat_analysis': {
                'n_total': 0,
                'n_sellable': 0,
                'n_outliers': 0,
                'n_used': 0,
                'expected_price': 0,
                'method': 'no_data',
            },
        }

    # Join with dogma attributes to get primary stat value
    primary_attrs = dogma_attrs[dogma_attrs['attribute_id'] == primary_attr_id][['item_id', 'value']].copy()
    primary_attrs = primary_attrs.rename(columns={'value': 'stat_value'})

    items_with_stats = matching_items.merge(primary_attrs, on='item_id', how='inner')

    # Join with prices
    items_with_prices = items_with_stats.merge(item_contracts, on='contract_id', how='inner')

    if items_with_prices.empty:
        return {
            'count': 0,
            'stats': np.array([]),
            'prices': np.array([]),
            'stat_analysis': {
                'n_total': 0,
                'n_sellable': 0,
                'n_outliers': 0,
                'n_used': 0,
                'expected_price': 0,
                'method': 'no_stats',
            },
        }

    stats = items_with_prices['stat_value'].values
    prices = items_with_prices['price'].values

    # Use stat-based price analysis
    stat_analysis = find_realistic_sale_price_stat_based(stats, prices, base_stat, max_stat)

    return {
        'count': len(stats),
        'stats': stats,
        'prices': prices,
        'stat_analysis': stat_analysis,
    }


def simulate_rolls(target: RollTarget, n_samples: int = NUM_SAMPLES) -> np.ndarray:
    """Simulate n_samples rolls and return array of rolled stats."""
    results = {}

    for muta_range in target.muta_ranges:
        # Uniform distribution between min and max multipliers
        rolls = np.random.uniform(muta_range.min_mult, muta_range.max_mult, n_samples)
        results[muta_range.attr_id] = rolls

    return results


def calc_dps_success_rate(target: RollTarget, damage_attr: int = ATTR_DAMAGE) -> tuple:
    """
    Calculate success rate for DPS-based modules (gyros, heatsinks, BCS).
    DPS = damage_mult / rof_mult (higher is better)
    """
    rolls = simulate_rolls(target)

    # Calculate DPS for each roll
    base_damage = target.base_stats.get(damage_attr, target.base_stats.get(213, 1.0))
    base_rof = target.base_stats.get(ATTR_ROF, 1.0)

    rolled_damage = base_damage * rolls[damage_attr if damage_attr in rolls else 213]
    rolled_rof = base_rof * rolls[ATTR_ROF]
    rolled_dps = rolled_damage / rolled_rof

    # Median DPS threshold (50th percentile)
    median_dps = np.median(rolled_dps)
    primary_success = rolled_dps >= median_dps

    # CPU catastrophe threshold (worst 10%)
    base_cpu = target.base_stats.get(ATTR_CPU, 1.0)
    rolled_cpu = base_cpu * rolls[ATTR_CPU]
    cpu_catastrophe_threshold = np.percentile(rolled_cpu, 90)  # Worst 10% = above 90th percentile
    secondary_ok = rolled_cpu < cpu_catastrophe_threshold

    combined_success = primary_success & secondary_ok
    success_rate = np.mean(combined_success)
    p_primary = np.mean(primary_success)
    p_secondary = np.mean(secondary_ok)

    return success_rate, p_primary, p_secondary, median_dps


def calc_web_success_rate(target: RollTarget) -> tuple:
    """
    Calculate success rate for stasis webifiers.
    Both web strength AND range need to be above median.
    """
    rolls = simulate_rolls(target)

    base_web = abs(target.base_stats.get(ATTR_VELOCITY_BONUS, -60))
    base_range = target.base_stats.get(ATTR_RANGE, 14000)

    # For webs, higher multiplier = stronger web (more velocity reduction)
    rolled_web = base_web * rolls[ATTR_VELOCITY_BONUS]
    rolled_range = base_range * rolls[ATTR_RANGE]

    median_web = np.median(rolled_web)
    median_range = np.median(rolled_range)

    # Both web AND range must be >= median
    primary_success = (rolled_web >= median_web) & (rolled_range >= median_range)

    # CPU catastrophe check
    base_cpu = target.base_stats.get(ATTR_CPU, 25)
    rolled_cpu = base_cpu * rolls[ATTR_CPU]
    cpu_catastrophe_threshold = np.percentile(rolled_cpu, 90)
    secondary_ok = rolled_cpu < cpu_catastrophe_threshold

    combined_success = primary_success & secondary_ok
    success_rate = np.mean(combined_success)
    p_primary = np.mean(primary_success)
    p_secondary = np.mean(secondary_ok)

    return success_rate, p_primary, p_secondary, f"Web>={median_web:.1f}%, Range>={median_range/1000:.1f}km"


def calc_shield_booster_success_rate(target: RollTarget) -> tuple:
    """
    Calculate success rate for shield boosters.
    Shield boost/s = boost_amount / duration
    """
    rolls = simulate_rolls(target)

    base_boost = target.base_stats.get(ATTR_SHIELD_BOOST, 256)
    base_duration = target.base_stats.get(ATTR_DURATION, 4000)

    rolled_boost = base_boost * rolls[ATTR_SHIELD_BOOST]
    rolled_duration = base_duration * rolls[ATTR_DURATION]
    rolled_boost_per_sec = rolled_boost / (rolled_duration / 1000)  # Convert ms to seconds

    median_boost_per_sec = np.median(rolled_boost_per_sec)
    primary_success = rolled_boost_per_sec >= median_boost_per_sec

    # Cap usage catastrophe check
    base_cap = target.base_stats.get(ATTR_CAP, 280)
    rolled_cap = base_cap * rolls[ATTR_CAP]
    cap_catastrophe_threshold = np.percentile(rolled_cap, 90)
    secondary_ok = rolled_cap < cap_catastrophe_threshold

    combined_success = primary_success & secondary_ok
    success_rate = np.mean(combined_success)
    p_primary = np.mean(primary_success)
    p_secondary = np.mean(secondary_ok)

    return success_rate, p_primary, p_secondary, median_boost_per_sec


def calc_plate_success_rate(target: RollTarget) -> tuple:
    """
    Calculate success rate for armor plates.
    HP needs to be above median, mass not catastrophic.
    """
    rolls = simulate_rolls(target)

    base_hp = target.base_stats.get(ATTR_ARMOR_HP, 6000)
    rolled_hp = base_hp * rolls[ATTR_ARMOR_HP]

    median_hp = np.median(rolled_hp)
    primary_success = rolled_hp >= median_hp

    # Mass catastrophe check
    base_mass = target.base_stats.get(ATTR_MASS, 2500000)
    rolled_mass = base_mass * rolls[ATTR_MASS]
    mass_catastrophe_threshold = np.percentile(rolled_mass, 90)
    secondary_ok = rolled_mass < mass_catastrophe_threshold

    combined_success = primary_success & secondary_ok
    success_rate = np.mean(combined_success)
    p_primary = np.mean(primary_success)
    p_secondary = np.mean(secondary_ok)

    return success_rate, p_primary, p_secondary, median_hp


def calc_ab_success_rate(target: RollTarget) -> tuple:
    """
    Calculate success rate for afterburners.
    Velocity bonus needs to be above median, cap not catastrophic.
    """
    rolls = simulate_rolls(target)

    base_velocity = target.base_stats.get(ATTR_VELOCITY_BONUS, 171)
    rolled_velocity = base_velocity * rolls[ATTR_VELOCITY_BONUS]

    median_velocity = np.median(rolled_velocity)
    primary_success = rolled_velocity >= median_velocity

    # Cap catastrophe check
    base_cap = target.base_stats.get(ATTR_CAP, 135)
    rolled_cap = base_cap * rolls[ATTR_CAP]
    cap_catastrophe_threshold = np.percentile(rolled_cap, 90)
    secondary_ok = rolled_cap < cap_catastrophe_threshold

    combined_success = primary_success & secondary_ok
    success_rate = np.mean(combined_success)
    p_primary = np.mean(primary_success)
    p_secondary = np.mean(secondary_ok)

    return success_rate, p_primary, p_secondary, median_velocity


def calc_dda_success_rate(target: RollTarget) -> tuple:
    """
    Calculate success rate for drone damage amplifiers.
    Damage bonus needs to be ABOVE BASE (sellable), CPU not catastrophic.

    A roll is sellable only if it improves upon the base item.
    """
    rolls = simulate_rolls(target)

    base_damage = target.base_stats.get(ATTR_DDA_DAMAGE, 23.8)
    rolled_damage = base_damage * rolls[ATTR_DDA_DAMAGE]

    # Sellable = rolled damage > base damage (improvement over unrolled item)
    primary_success = rolled_damage > base_damage

    # CPU catastrophe check
    base_cpu = target.base_stats.get(ATTR_CPU, 45)
    rolled_cpu = base_cpu * rolls[ATTR_CPU]
    cpu_catastrophe_threshold = np.percentile(rolled_cpu, 90)
    secondary_ok = rolled_cpu < cpu_catastrophe_threshold

    combined_success = primary_success & secondary_ok
    success_rate = np.mean(combined_success)
    p_primary = np.mean(primary_success)
    p_secondary = np.mean(secondary_ok)

    return success_rate, p_primary, p_secondary, base_damage


def calc_armor_rep_success_rate(target: RollTarget) -> tuple:
    """
    Calculate success rate for armor repairers.
    Armor rep/s = repair_amount / duration
    """
    rolls = simulate_rolls(target)

    base_rep = target.base_stats.get(84, 240)  # attr 84 = armor repair amount
    base_duration = target.base_stats.get(ATTR_DURATION, 6000)

    rolled_rep = base_rep * rolls[84]
    rolled_duration = base_duration * rolls[ATTR_DURATION]
    rolled_rep_per_sec = rolled_rep / (rolled_duration / 1000)

    median_rep_per_sec = np.median(rolled_rep_per_sec)
    primary_success = rolled_rep_per_sec >= median_rep_per_sec

    # Cap usage catastrophe check
    base_cap = target.base_stats.get(ATTR_CAP, 160)
    rolled_cap = base_cap * rolls[ATTR_CAP]
    cap_catastrophe_threshold = np.percentile(rolled_cap, 90)
    secondary_ok = rolled_cap < cap_catastrophe_threshold

    combined_success = primary_success & secondary_ok
    success_rate = np.mean(combined_success)
    p_primary = np.mean(primary_success)
    p_secondary = np.mean(secondary_ok)

    return success_rate, p_primary, p_secondary, median_rep_per_sec


def calc_shield_extender_success_rate(target: RollTarget) -> tuple:
    """
    Calculate success rate for shield extenders.
    Shield HP needs to be above base, CPU/PG not catastrophic.
    """
    rolls = simulate_rolls(target)

    base_hp = target.base_stats.get(ATTR_SHIELD_CAP, 2750)
    rolled_hp = base_hp * rolls[ATTR_SHIELD_CAP]

    # Sellable = shield HP > base
    primary_success = rolled_hp > base_hp

    # CPU catastrophe check
    base_cpu = target.base_stats.get(ATTR_CPU, 35)
    rolled_cpu = base_cpu * rolls[ATTR_CPU]
    cpu_catastrophe_threshold = np.percentile(rolled_cpu, 90)
    secondary_ok = rolled_cpu < cpu_catastrophe_threshold

    combined_success = primary_success & secondary_ok
    success_rate = np.mean(combined_success)
    p_primary = np.mean(primary_success)
    p_secondary = np.mean(secondary_ok)

    return success_rate, p_primary, p_secondary, base_hp


def calc_cap_battery_success_rate(target: RollTarget) -> tuple:
    """
    Calculate success rate for cap batteries.
    Cap capacity needs to be above base, CPU not catastrophic.
    """
    rolls = simulate_rolls(target)

    base_cap = target.base_stats.get(ATTR_CAP_CAPACITY, 1820)
    rolled_cap = base_cap * rolls[ATTR_CAP_CAPACITY]

    # Sellable = cap capacity > base
    primary_success = rolled_cap > base_cap

    # CPU catastrophe check
    base_cpu = target.base_stats.get(ATTR_CPU, 40)
    rolled_cpu = base_cpu * rolls[ATTR_CPU]
    cpu_catastrophe_threshold = np.percentile(rolled_cpu, 90)
    secondary_ok = rolled_cpu < cpu_catastrophe_threshold

    combined_success = primary_success & secondary_ok
    success_rate = np.mean(combined_success)
    p_primary = np.mean(primary_success)
    p_secondary = np.mean(secondary_ok)

    return success_rate, p_primary, p_secondary, base_cap


def calc_bcs_success_rate(target: RollTarget) -> tuple:
    """
    Calculate success rate for ballistic control systems.
    DPS = missile_damage_mult / rof_mult (higher is better)
    Sellable = DPS > base DPS
    """
    rolls = simulate_rolls(target)

    base_damage = target.base_stats.get(ATTR_MISSILE_DAMAGE, 1.1)
    base_rof = target.base_stats.get(ATTR_ROF, 0.895)
    base_dps = base_damage / base_rof

    rolled_damage = base_damage * rolls[ATTR_MISSILE_DAMAGE]
    rolled_rof = base_rof * rolls[ATTR_ROF]
    rolled_dps = rolled_damage / rolled_rof

    # Sellable = DPS > base DPS
    primary_success = rolled_dps > base_dps

    # CPU catastrophe check
    base_cpu = target.base_stats.get(ATTR_CPU, 40)
    rolled_cpu = base_cpu * rolls[ATTR_CPU]
    cpu_catastrophe_threshold = np.percentile(rolled_cpu, 90)
    secondary_ok = rolled_cpu < cpu_catastrophe_threshold

    combined_success = primary_success & secondary_ok
    success_rate = np.mean(combined_success)
    p_primary = np.mean(primary_success)
    p_secondary = np.mean(secondary_ok)

    return success_rate, p_primary, p_secondary, base_dps


def calc_warp_disruptor_success_rate(target: RollTarget) -> tuple:
    """
    Calculate success rate for warp disruptors.
    Range needs to be above base, CPU not catastrophic.
    """
    rolls = simulate_rolls(target)

    base_range = target.base_stats.get(ATTR_RANGE, 30000)
    rolled_range = base_range * rolls[ATTR_RANGE]

    # Sellable = range > base
    primary_success = rolled_range > base_range

    # CPU catastrophe check
    base_cpu = target.base_stats.get(ATTR_CPU, 38)
    rolled_cpu = base_cpu * rolls[ATTR_CPU]
    cpu_catastrophe_threshold = np.percentile(rolled_cpu, 90)
    secondary_ok = rolled_cpu < cpu_catastrophe_threshold

    combined_success = primary_success & secondary_ok
    success_rate = np.mean(combined_success)
    p_primary = np.mean(primary_success)
    p_secondary = np.mean(secondary_ok)

    return success_rate, p_primary, p_secondary, base_range


def calc_damage_control_success_rate(target: RollTarget) -> tuple:
    """
    Calculate success rate for damage controls.
    For now, just check CPU isn't catastrophic (hull resists are complex).
    Assume 50% base success rate for simplicity.
    """
    rolls = simulate_rolls(target)

    # CPU catastrophe check
    base_cpu = target.base_stats.get(ATTR_CPU, 30)
    rolled_cpu = base_cpu * rolls[ATTR_CPU]
    cpu_catastrophe_threshold = np.percentile(rolled_cpu, 90)
    secondary_ok = rolled_cpu < cpu_catastrophe_threshold

    # Assume ~50% primary success (resist rolls are complex)
    primary_success = np.random.random(len(rolled_cpu)) > 0.5

    combined_success = primary_success & secondary_ok
    success_rate = np.mean(combined_success)
    p_primary = np.mean(primary_success)
    p_secondary = np.mean(secondary_ok)

    return success_rate, p_primary, p_secondary, 0.5


def calc_dps_stat_based_success_rate(target: RollTarget, damage_attr: int = ATTR_DAMAGE) -> tuple:
    """
    Calculate success rate for DPS modules using stat > base threshold.
    DPS = damage_mult / rof_mult (higher is better)
    Sellable = DPS > base DPS
    """
    rolls = simulate_rolls(target)

    base_damage = target.base_stats.get(damage_attr, 1.1)
    base_rof = target.base_stats.get(ATTR_ROF, 0.9)
    base_dps = base_damage / base_rof

    rolled_damage = base_damage * rolls[damage_attr]
    rolled_rof = base_rof * rolls[ATTR_ROF]
    rolled_dps = rolled_damage / rolled_rof

    # Sellable = DPS > base DPS (improvement over unrolled item)
    primary_success = rolled_dps > base_dps

    # CPU catastrophe check
    base_cpu = target.base_stats.get(ATTR_CPU, 30)
    rolled_cpu = base_cpu * rolls[ATTR_CPU]
    cpu_catastrophe_threshold = np.percentile(rolled_cpu, 90)
    secondary_ok = rolled_cpu < cpu_catastrophe_threshold

    combined_success = primary_success & secondary_ok
    success_rate = np.mean(combined_success)
    p_primary = np.mean(primary_success)
    p_secondary = np.mean(secondary_ok)

    return success_rate, p_primary, p_secondary, base_dps


def calc_web_stat_based_success_rate(target: RollTarget) -> tuple:
    """
    Calculate success rate for webs using stat > base threshold.
    Both web strength AND range need to be above base to be sellable.
    """
    rolls = simulate_rolls(target)

    base_web = abs(target.base_stats.get(ATTR_VELOCITY_BONUS, -60))
    base_range = target.base_stats.get(ATTR_RANGE, 14000)

    rolled_web = base_web * rolls[ATTR_VELOCITY_BONUS]
    rolled_range = base_range * rolls[ATTR_RANGE]

    # Sellable = web > base AND range > base
    primary_success = (rolled_web > base_web) & (rolled_range > base_range)

    # CPU catastrophe check
    base_cpu = target.base_stats.get(ATTR_CPU, 25)
    rolled_cpu = base_cpu * rolls[ATTR_CPU]
    cpu_catastrophe_threshold = np.percentile(rolled_cpu, 90)
    secondary_ok = rolled_cpu < cpu_catastrophe_threshold

    combined_success = primary_success & secondary_ok
    success_rate = np.mean(combined_success)
    p_primary = np.mean(primary_success)
    p_secondary = np.mean(secondary_ok)

    return success_rate, p_primary, p_secondary, f"Web>{base_web:.1f}%, Range>{base_range/1000:.1f}km"


def calc_shield_booster_stat_based_success_rate(target: RollTarget) -> tuple:
    """
    Calculate success rate for shield boosters using stat > base threshold.
    Shield boost/s = boost_amount / duration
    Sellable = boost/s > base boost/s
    """
    rolls = simulate_rolls(target)

    base_boost = target.base_stats.get(ATTR_SHIELD_BOOST, 256)
    base_duration = target.base_stats.get(ATTR_DURATION, 4000)
    base_boost_per_sec = base_boost / (base_duration / 1000)

    rolled_boost = base_boost * rolls[ATTR_SHIELD_BOOST]
    rolled_duration = base_duration * rolls[ATTR_DURATION]
    rolled_boost_per_sec = rolled_boost / (rolled_duration / 1000)

    # Sellable = boost/s > base boost/s
    primary_success = rolled_boost_per_sec > base_boost_per_sec

    # Cap usage catastrophe check
    base_cap = target.base_stats.get(ATTR_CAP, 280)
    rolled_cap = base_cap * rolls[ATTR_CAP]
    cap_catastrophe_threshold = np.percentile(rolled_cap, 90)
    secondary_ok = rolled_cap < cap_catastrophe_threshold

    combined_success = primary_success & secondary_ok
    success_rate = np.mean(combined_success)
    p_primary = np.mean(primary_success)
    p_secondary = np.mean(secondary_ok)

    return success_rate, p_primary, p_secondary, base_boost_per_sec


def calc_plate_stat_based_success_rate(target: RollTarget) -> tuple:
    """
    Calculate success rate for armor plates using stat > base threshold.
    HP needs to be above base, mass not catastrophic.
    """
    rolls = simulate_rolls(target)

    base_hp = target.base_stats.get(ATTR_ARMOR_HP, 6000)
    rolled_hp = base_hp * rolls[ATTR_ARMOR_HP]

    # Sellable = HP > base HP
    primary_success = rolled_hp > base_hp

    # Mass catastrophe check
    base_mass = target.base_stats.get(ATTR_MASS, 2500000)
    rolled_mass = base_mass * rolls[ATTR_MASS]
    mass_catastrophe_threshold = np.percentile(rolled_mass, 90)
    secondary_ok = rolled_mass < mass_catastrophe_threshold

    combined_success = primary_success & secondary_ok
    success_rate = np.mean(combined_success)
    p_primary = np.mean(primary_success)
    p_secondary = np.mean(secondary_ok)

    return success_rate, p_primary, p_secondary, base_hp


def calc_ab_stat_based_success_rate(target: RollTarget) -> tuple:
    """
    Calculate success rate for afterburners using stat > base threshold.
    Velocity bonus needs to be above base, cap not catastrophic.
    """
    rolls = simulate_rolls(target)

    base_velocity = target.base_stats.get(ATTR_VELOCITY_BONUS, 145)
    rolled_velocity = base_velocity * rolls[ATTR_VELOCITY_BONUS]

    # Sellable = velocity > base
    primary_success = rolled_velocity > base_velocity

    # Cap catastrophe check
    base_cap = target.base_stats.get(ATTR_CAP, 135)
    rolled_cap = base_cap * rolls[ATTR_CAP]
    cap_catastrophe_threshold = np.percentile(rolled_cap, 90)
    secondary_ok = rolled_cap < cap_catastrophe_threshold

    combined_success = primary_success & secondary_ok
    success_rate = np.mean(combined_success)
    p_primary = np.mean(primary_success)
    p_secondary = np.mean(secondary_ok)

    return success_rate, p_primary, p_secondary, base_velocity


def get_success_rate(key: str, target: RollTarget) -> tuple:
    """Route to appropriate success rate calculator based on module type."""
    module_type = target.module_type

    if module_type == 'dps':
        return calc_dps_stat_based_success_rate(target, ATTR_DAMAGE)
    elif module_type == 'bcs':
        return calc_bcs_success_rate(target)
    elif module_type == 'web':
        return calc_web_stat_based_success_rate(target)
    elif module_type == 'shield_booster':
        return calc_shield_booster_stat_based_success_rate(target)
    elif module_type == 'armor_plate':
        return calc_plate_stat_based_success_rate(target)
    elif module_type == 'afterburner':
        return calc_ab_stat_based_success_rate(target)
    elif module_type == 'dda':
        return calc_dda_success_rate(target)
    elif module_type == 'shield_extender':
        return calc_shield_extender_success_rate(target)
    elif module_type == 'cap_battery':
        return calc_cap_battery_success_rate(target)
    elif module_type == 'warp_disruptor':
        return calc_warp_disruptor_success_rate(target)
    elif module_type == 'damage_control':
        return calc_damage_control_success_rate(target)
    elif module_type == 'armor_repairer':
        return calc_armor_rep_success_rate(target)
    else:
        # Default: DPS calculation
        return calc_dps_stat_based_success_rate(target, ATTR_DAMAGE)


def format_isk(value: float) -> str:
    """Format ISK value."""
    # Handle negative values
    if value < 0:
        return f"-{format_isk(abs(value))}"

    if value >= 1_000_000_000:
        return f"{value/1_000_000_000:.2f}B"
    elif value >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    elif value >= 1_000:
        return f"{value/1_000:.1f}K"
    return f"{value:.0f}"


def format_stat(value: float, module_type: str, attr_id: int = None) -> str:
    """Format stat value based on module type."""
    if module_type in ('dps', 'bcs'):
        # DPS multipliers: format as x.xxx
        return f"{value:.3f}x"
    elif module_type == 'dda':
        # DDA: percentage bonus
        return f"{value:.2f}%"
    elif module_type == 'shield_extender':
        # Shield HP
        return f"{value:.0f} HP"
    elif module_type == 'cap_battery':
        # Cap capacity GJ
        return f"{value:.0f} GJ"
    elif module_type == 'web':
        # Web: velocity factor (stored as negative)
        if attr_id == ATTR_VELOCITY_BONUS:
            return f"{abs(value):.1f}%"
        elif attr_id == ATTR_RANGE:
            return f"{value/1000:.1f}km"
        return f"{value:.2f}"
    elif module_type == 'armor_plate':
        # Armor HP
        return f"{value:.0f} HP"
    elif module_type == 'shield_booster':
        # Shield boost/s
        return f"{value:.1f} HP/s"
    elif module_type == 'afterburner':
        # Velocity bonus %
        return f"{value:.1f}%"
    elif module_type == 'warp_disruptor':
        # Range in km
        return f"{value/1000:.1f}km"
    elif module_type == 'damage_control':
        return f"{value:.2f}"
    else:
        # Default: just show the number
        return f"{value:.2f}"


def calculate_profit_probability(
    bankroll: float,
    roll_cost: float,
    success_rate: float,
    sale_price: float,
) -> dict:
    """
    Calculate probability of being profitable with a given bankroll.

    With bankroll B and roll_cost C, we can do N = floor(B / C) rolls.
    Each roll is a Bernoulli trial:
      - Success (prob p): gain (sale_price - roll_cost)
      - Failure (prob 1-p): lose roll_cost

    After N rolls with K successes:
      profit = K * sale_price - N * roll_cost

    For profit > 0: K > N * roll_cost / sale_price

    Returns probability metrics and distribution info.
    """
    n_rolls = int(bankroll // roll_cost)

    if n_rolls == 0:
        return {
            'n_rolls': 0,
            'prob_profitable': 0,
            'prob_breakeven': 0,
            'expected_profit': -bankroll,
            'profit_at_5pct': -bankroll,
            'profit_at_50pct': -bankroll,
            'profit_at_95pct': -bankroll,
            'min_successes_needed': float('inf'),
        }

    # Minimum successes needed for profit > 0
    # K * sale_price > N * roll_cost
    # K > N * roll_cost / sale_price
    breakeven_k = n_rolls * roll_cost / sale_price if sale_price > 0 else float('inf')
    min_k_for_profit = int(np.ceil(breakeven_k))

    # Use binomial distribution
    binom = stats.binom(n=n_rolls, p=success_rate)

    # P(profit > 0) = P(K >= min_k_for_profit) = 1 - P(K < min_k_for_profit)
    prob_profitable = 1 - binom.cdf(min_k_for_profit - 1) if min_k_for_profit <= n_rolls else 0

    # P(breakeven or better) = P(K >= ceil(breakeven_k))
    prob_breakeven = prob_profitable  # Same threshold

    # Expected profit = E[K] * sale_price - N * roll_cost
    expected_successes = n_rolls * success_rate
    expected_profit = expected_successes * sale_price - n_rolls * roll_cost

    # Calculate profit distribution percentiles
    # For each possible K (0 to n_rolls), calculate profit
    k_values = np.arange(0, n_rolls + 1)
    profits = k_values * sale_price - n_rolls * roll_cost
    probabilities = binom.pmf(k_values)

    # Find percentiles (5th, 50th, 95th)
    cumprob = np.cumsum(probabilities)

    def find_percentile_profit(target_pct):
        idx = np.searchsorted(cumprob, target_pct / 100)
        idx = min(idx, len(profits) - 1)
        return profits[idx]

    profit_5pct = find_percentile_profit(5)
    profit_50pct = find_percentile_profit(50)
    profit_95pct = find_percentile_profit(95)

    # Probability of total loss (0 successes)
    prob_total_loss = binom.pmf(0)

    return {
        'n_rolls': n_rolls,
        'prob_profitable': prob_profitable,
        'prob_breakeven': prob_breakeven,
        'expected_profit': expected_profit,
        'profit_at_5pct': profit_5pct,
        'profit_at_50pct': profit_50pct,
        'profit_at_95pct': profit_95pct,
        'min_successes_needed': min_k_for_profit,
        'expected_successes': expected_successes,
        'prob_total_loss': prob_total_loss,
        'breakeven_k': breakeven_k,
    }


def calculate_roi(roll_cost: float, success_rate: float, sale_price: float) -> dict:
    """Calculate expected value and ROI per roll."""

    # Expected value per roll = success_rate * sale_price
    expected_value = success_rate * sale_price

    # Expected profit = expected_value - roll_cost
    expected_profit = expected_value - roll_cost

    # ROI = expected_profit / roll_cost
    roi = (expected_profit / roll_cost) * 100 if roll_cost > 0 else 0

    # Break-even success rate
    breakeven_rate = roll_cost / sale_price if sale_price > 0 else 1

    # Rolls per success on average
    rolls_per_success = 1 / success_rate if success_rate > 0 else float('inf')

    return {
        'roll_cost': roll_cost,
        'expected_value': expected_value,
        'expected_profit': expected_profit,
        'roi_pct': roi,
        'breakeven_rate': breakeven_rate,
        'rolls_per_success': rolls_per_success,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calculate ROI for rolling mutated modules"
    )
    parser.add_argument(
        '--target', '-t',
        choices=list(ROLL_TARGETS.keys()),
        default=None,
        help='Specific target to analyze (default: all)'
    )
    parser.add_argument(
        '--bankroll', '-b',
        type=float,
        default=1_000_000_000,
        help='Bankroll in ISK for risk analysis (default: 1B)'
    )
    parser.add_argument(
        '--confidence', '-c',
        type=float,
        default=95.0,
        help='Minimum probability of profit %% for "safe" items (default: 95)'
    )
    parser.add_argument(
        '--max-sells', '-m',
        type=int,
        default=5,
        help='Maximum number of successful sales to reach profit (default: 5)'
    )

    args = parser.parse_args()
    bankroll = args.bankroll
    confidence_threshold = args.confidence / 100
    max_sells = args.max_sells

    print("=" * 70)
    print("EVE Online Mutated Module ROI Calculator")
    print("=" * 70)
    print(f"\nUsing {NUM_SAMPLES:,} Monte Carlo samples per item")
    print(f"Bankroll: {format_isk(bankroll)} | Confidence: {args.confidence:.0f}% | Max sells: {max_sells}")

    # Get targets to analyze
    if args.target:
        targets = {args.target: ROLL_TARGETS[args.target]}
    else:
        targets = ROLL_TARGETS

    # Collect all type IDs we need prices for
    all_type_ids = set()
    for target in targets.values():
        all_type_ids.add(target.base_type_id)
        all_type_ids.add(target.muta_type_id)

    # Get market prices
    print("\nFetching Jita market prices...")
    prices = get_jita_prices(list(all_type_ids))

    # Download contract data
    archive_path = download_contract_data()

    # Analyze each target
    results = []

    print("\nAnalyzing each item...")
    for key, target in targets.items():
        print(f"  {target.name}...")

        # Get costs first (needed for price analysis)
        base_price = prices[target.base_type_id]['min']
        muta_price = prices[target.muta_type_id]['min']
        roll_cost = base_price + muta_price

        # Use stat-based analysis for ALL items
        if True:  # All items now use stat-based analysis
            # Get primary attribute info from muta_ranges
            primary_range = target.muta_ranges[0]  # First range is primary stat
            base_stat = target.base_stats.get(primary_range.attr_id, 0)
            max_stat = base_stat * primary_range.max_mult

            # Analyze using stat-based method (ALL items with this mutaplasmid)
            dist = analyze_contracts_stat_based(
                archive_path,
                target.muta_type_id,
                primary_range.attr_id,
                base_stat,
                max_stat,
            )

            if dist['count'] == 0:
                print(f"    No contracts found, skipping...")
                continue

            stat_analysis = dist['stat_analysis']
            realistic_sale_price = stat_analysis['expected_price']

            # Calculate success rate: P(stat > base) * P(secondary OK)
            success_rate, p_primary, p_secondary, threshold = get_success_rate(key, target)

            # Store stat-based analysis for display
            price_analysis = {
                'n_total': stat_analysis['n_total'],
                'n_sellable': stat_analysis['n_sellable'],
                'n_outliers': stat_analysis['n_outliers'],
                'n_used': stat_analysis['n_used'],
                'expected_price': stat_analysis['expected_price'],
                'method': stat_analysis['method'],
                'midpoint_stat': stat_analysis.get('midpoint_stat', 0),
                'slope': stat_analysis.get('slope', 0),
                'anchor_stat': stat_analysis.get('anchor_stat', 0),
                'anchor_price': stat_analysis.get('anchor_price', 0),
                'base_stat': base_stat,
                'max_stat': max_stat,
                'coverage_pct': stat_analysis.get('coverage_pct', 0),
                'data_min_stat': stat_analysis.get('data_min_stat', 0),
                'data_max_stat': stat_analysis.get('data_max_stat', 0),
                'stat_based': True,
                'module_type': target.module_type,
                'primary_attr_id': primary_range.attr_id,
            }

        # Calculate ROI: expected_profit = success_rate * sale_price - roll_cost
        roi = calculate_roi(
            roll_cost,
            success_rate,
            realistic_sale_price
        )

        # Calculate risk metrics with given bankroll
        risk = calculate_profit_probability(
            bankroll,
            roll_cost,
            success_rate,
            realistic_sale_price
        )

        results.append({
            'key': key,
            'target': target,
            'base_price': base_price,
            'muta_price': muta_price,
            'roll_cost': roll_cost,
            'distribution': dist,
            'price_analysis': price_analysis,
            'roi': roi,
            'risk': risk,
            'success_rate': success_rate,
            'p_primary': p_primary,
            'p_secondary': p_secondary,
            'threshold': threshold,
            'sale_price': realistic_sale_price,
        })

    # Sort by expected profit per roll
    results.sort(key=lambda x: x['roi']['expected_profit'], reverse=True)

    # Display results
    print("\n" + "=" * 70)
    print("ROI ANALYSIS RESULTS (sorted by Profit/Roll)")
    print("=" * 70)

    for r in results:
        target = r['target']
        dist = r['distribution']
        pa = r['price_analysis']
        roi = r['roi']

        print(f"\n{'-' * 70}")
        print(f"  {target.name}")
        print(f"{'-' * 70}")

        print(f"\n  COSTS:")
        print(f"    Base item:   {format_isk(r['base_price']):>12}  ({target.base_name})")
        print(f"    Mutaplasmid: {format_isk(r['muta_price']):>12}  ({target.muta_name})")
        print(f"    Roll cost:   {format_isk(r['roll_cost']):>12}")

        # Stat-based analysis display (now used for all items)
        module_type = pa.get('module_type', 'dps')
        attr_id = pa.get('primary_attr_id', ATTR_DAMAGE)

        print(f"\n  CONTRACT ANALYSIS ({dist['count']} contracts, stat-based):")
        base_fmt = format_stat(pa['base_stat'], module_type, attr_id)
        max_fmt = format_stat(pa['max_stat'], module_type, attr_id)
        mid_fmt = format_stat(pa['midpoint_stat'], module_type, attr_id)
        print(f"    Stat range:    [{base_fmt} -> {max_fmt}]")
        print(f"    Midpoint:      {mid_fmt}")
        print(f"    Total items:               {pa['n_total']:>3}")
        print(f"    Below base (unsellable):   {pa['n_total'] - pa['n_sellable']:>3} (discarded)")
        print(f"    Price outliers (IQR):      {pa['n_outliers']:>3} (discarded)")
        print(f"    Used for regression:       {pa['n_used']:>3}")
        print(f"    Method:        {pa['method']}")
        if pa['method'] == 'constrained_regression':
            anchor_fmt = format_stat(pa['anchor_stat'], module_type, attr_id)
            print(f"    Anchor point:  ({anchor_fmt}, {format_isk(pa['anchor_price'])})")
            print(f"    Slope:         {format_isk(pa['slope'])}/unit stat")
            # Show data coverage
            coverage = pa.get('coverage_pct', 0)
            data_min = pa.get('data_min_stat', 0)
            data_max = pa.get('data_max_stat', 0)
            data_min_fmt = format_stat(data_min, module_type, attr_id)
            data_max_fmt = format_stat(data_max, module_type, attr_id)
            coverage_warning = " ** LOW CONFIDENCE **" if coverage < 50 else ""
            print(f"    Data coverage: [{data_min_fmt} -> {data_max_fmt}] = {coverage:.0f}%{coverage_warning}")
        print(f"    Exp. price at midpoint: {format_isk(pa['expected_price']):>12}")

        print(f"\n  SUCCESS PROBABILITY (Monte Carlo):")
        print(f"    Primary:     {target.primary_desc}")
        print(f"                 {r['p_primary']*100:>10.1f}%")
        print(f"    Secondary:   {target.secondary_desc}")
        print(f"                 {r['p_secondary']*100:>10.1f}%")
        print(f"    Sellable:    {r['success_rate']*100:>10.1f}%  (stat > base + OK secondary)")

        print(f"\n  ROI ANALYSIS:")
        print(f"    Exp. sale:     {format_isk(pa['expected_price']):>12}  (regression midpoint)")
        print(f"    Exp. value:    {format_isk(roi['expected_value']):>12}  per roll")
        print(f"    Exp. profit:   {format_isk(roi['expected_profit']):>12}  per roll")
        print(f"    ROI:           {roi['roi_pct']:>11.1f}%")
        print(f"    Rolls/success: {roi['rolls_per_success']:>11.1f}")

        # Verdict
        if roi['roi_pct'] > 100:
            verdict = "EXCELLENT"
        elif roi['roi_pct'] > 50:
            verdict = "GOOD"
        elif roi['roi_pct'] > 0:
            verdict = "MARGINAL"
        else:
            verdict = "AVOID"

        # Add coverage warning to verdict if applicable
        coverage = pa.get('coverage_pct', 100)
        if coverage < 50:
            verdict += " (LOW CONFIDENCE)"

        print(f"\n  VERDICT: {verdict}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - RANKED BY PROFIT PER ROLL")
    print("=" * 70)
    print(f"\n{'Rank':<4} {'Profit/Roll':>12} {'ROI':>8} {'Success':>8} {'Coverage':>8} {'Roll Cost':>12}  Item")
    print("-" * 90)

    for i, r in enumerate(results[:10], 1):
        roi = r['roi']
        pa = r['price_analysis']
        profit = roi['expected_profit']
        profit_str = format_isk(profit) if profit >= 0 else f"-{format_isk(abs(profit))}"
        coverage = pa.get('coverage_pct', 100)
        coverage_str = f"{coverage:.0f}%"
        low_conf = "*" if coverage < 50 else ""
        print(f"{i:<4} {profit_str:>12} {roi['roi_pct']:>7.1f}% {r['success_rate']*100:>7.1f}% {coverage_str:>8}{low_conf} {format_isk(r['roll_cost']):>11}  {r['target'].name}")

    # Legend for low confidence
    if any(r['price_analysis'].get('coverage_pct', 100) < 50 for r in results[:10]):
        print("\n* = Low confidence (data coverage < 50%)")

    # Risk-Adjusted Analysis
    print("\n" + "=" * 70)
    print(f"RISK ANALYSIS - {format_isk(bankroll)} BANKROLL")
    print("=" * 70)
    print(f"\nItems with >= {args.confidence:.0f}% probability of profit AND <= {max_sells} sales to profit:")
    print(f"(Practical constraint: you don't want to roll/sell hundreds of items)\n")

    # Filter to "safe" items: meet confidence threshold AND need <= max_sells successes
    safe_items = [
        r for r in results
        if r['risk']['prob_profitable'] >= confidence_threshold
        and r['risk']['min_successes_needed'] <= max_sells
    ]

    # Sort safe items by expected total profit (not per-roll)
    safe_items.sort(key=lambda x: x['risk']['expected_profit'], reverse=True)

    if not safe_items:
        print(f"  No items meet both criteria with {format_isk(bankroll)} bankroll.")
        print(f"  Try: increasing bankroll (-b), lowering confidence (-c), or raising max-sells (-m)")
    else:
        print(f"{'Rank':<4} {'P(Profit)':>9} {'Rolls':>6} {'Need':>5} {'E[Profit]':>12} {'5th%':>10} {'Median':>10} {'ROI/Roll':>9}  Item")
        print("-" * 105)

        for i, r in enumerate(safe_items[:15], 1):
            risk = r['risk']
            roi = r['roi']
            prob_pct = risk['prob_profitable'] * 100
            n_rolls = risk['n_rolls']
            need = risk['min_successes_needed']
            exp_profit = risk['expected_profit']
            p5 = risk['profit_at_5pct']
            p50 = risk['profit_at_50pct']

            print(f"{i:<4} {prob_pct:>8.1f}% {n_rolls:>6} {need:>5} {format_isk(exp_profit):>12} {format_isk(p5):>10} {format_isk(p50):>10} {roi['roi_pct']:>8.1f}%  {r['target'].name}")

        # Show best "safe" item details
        best = safe_items[0]
        risk = best['risk']
        print(f"\n  RECOMMENDED: {best['target'].name}")
        print(f"  With {format_isk(bankroll)} you can do {risk['n_rolls']} rolls")
        print(f"  Need {risk['min_successes_needed']}/{risk['n_rolls']} successes to profit (breakeven: {risk['breakeven_k']:.2f})")
        print(f"  Probability of profit: {risk['prob_profitable']*100:.1f}%")
        print(f"  Expected profit: {format_isk(risk['expected_profit'])}")
        print(f"  Worst case (5th pct): {format_isk(risk['profit_at_5pct'])}")
        print(f"  Median outcome: {format_isk(risk['profit_at_50pct'])}")
        print(f"  Best case (95th pct): {format_isk(risk['profit_at_95pct'])}")

    # Show items that are safe but need too many sells (impractical)
    too_many_sells = [
        r for r in results
        if r['risk']['prob_profitable'] >= confidence_threshold
        and r['risk']['min_successes_needed'] > max_sells
    ]
    too_many_sells.sort(key=lambda x: x['risk']['expected_profit'], reverse=True)

    if too_many_sells:
        print(f"\n" + "-" * 70)
        print(f"HIGH CONFIDENCE BUT TOO MANY SELLS (need > {max_sells} sales):")
        print("-" * 70)
        print(f"{'P(Profit)':>9} {'Rolls':>6} {'Need':>5} {'ROI/Roll':>9}  Item")

        for r in too_many_sells[:5]:
            risk = r['risk']
            roi = r['roi']
            print(f"{risk['prob_profitable']*100:>8.1f}% {risk['n_rolls']:>6} {risk['min_successes_needed']:>5} {roi['roi_pct']:>8.1f}%  {r['target'].name}")

        print(f"\n  These are statistically safe but require too many rolls/sales.")
        print(f"  Use -m to increase max-sells if you have time for high-volume trading.")

    # Show risky high-ROI items (low confidence)
    risky_items = [
        r for r in results
        if r['risk']['prob_profitable'] < confidence_threshold
        and r['roi']['roi_pct'] > 50
        and r['risk']['min_successes_needed'] <= max_sells
    ]
    risky_items.sort(key=lambda x: x['roi']['expected_profit'], reverse=True)

    if risky_items:
        print(f"\n" + "-" * 70)
        print(f"HIGH ROI BUT RISKY (< {args.confidence:.0f}% confidence with {format_isk(bankroll)}):")
        print("-" * 70)
        print(f"{'P(Profit)':>9} {'Rolls':>6} {'Need':>5} {'ROI/Roll':>9} {'Profit/Roll':>12}  Item")

        for r in risky_items[:5]:
            risk = r['risk']
            roi = r['roi']
            print(f"{risk['prob_profitable']*100:>8.1f}% {risk['n_rolls']:>6} {risk['min_successes_needed']:>5} {roi['roi_pct']:>8.1f}% {format_isk(roi['expected_profit']):>12}  {r['target'].name}")

        print(f"\n  These items have great ROI but you can't do enough rolls with {format_isk(bankroll)}")
        print(f"  to reliably overcome RNG variance. Consider a larger bankroll.")


if __name__ == '__main__':
    main()
