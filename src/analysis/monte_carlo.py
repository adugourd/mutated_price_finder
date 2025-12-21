"""
Monte Carlo simulation for mutaplasmid roll success rates.

Simulates mutation rolls and calculates success probabilities
for different module types.
"""

from typing import Optional, Callable
from dataclasses import dataclass

import numpy as np

from src.config.loader import RollTarget, load_attributes, load_constants

# Load configuration
_attributes = load_attributes()
_constants = load_constants()

# Attribute IDs
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
ATTR_DDA_DAMAGE = _attributes['dda_damage']
ATTR_SHIELD_CAP = _attributes['shield_cap']
ATTR_CAP_CAPACITY = _attributes['cap_capacity']
ATTR_MISSILE_DAMAGE = _attributes['missile_damage']

# Simulation parameters
NUM_SAMPLES = _constants['simulation']['num_samples']
CATASTROPHE_PERCENTILE = _constants['analysis']['catastrophe_percentile']


@dataclass
class SuccessResult:
    """Result of success rate calculation."""
    success_rate: float
    p_primary: float
    p_secondary: float
    threshold: any  # Type varies by module type
    base_stat: Optional[float] = None


def simulate_rolls(target: RollTarget, n_samples: int = NUM_SAMPLES) -> dict[int, np.ndarray]:
    """
    Simulate n_samples rolls and return arrays of rolled stats.

    Args:
        target: Roll target configuration
        n_samples: Number of Monte Carlo samples

    Returns:
        Dict mapping attribute_id -> array of rolled multipliers
    """
    results = {}

    for muta_range in target.muta_ranges:
        # Uniform distribution between min and max multipliers
        rolls = np.random.uniform(muta_range.min_mult, muta_range.max_mult, n_samples)
        results[muta_range.attr_id] = rolls

    return results


def calc_dps_success_rate(target: RollTarget, damage_attr: int = ATTR_DAMAGE) -> SuccessResult:
    """
    Calculate success rate for DPS-based modules (gyros, heatsinks, mag stabs).

    DPS = damage_mult / rof_mult (higher is better)
    Primary: DPS >= median
    Secondary: CPU not in worst 10%
    """
    rolls = simulate_rolls(target)

    # Get the damage attribute (could be ATTR_DAMAGE or ATTR_MISSILE_DAMAGE for BCS)
    actual_damage_attr = damage_attr if damage_attr in rolls else ATTR_MISSILE_DAMAGE

    base_damage = target.base_stats.get(actual_damage_attr, 1.0)
    base_rof = target.base_stats.get(ATTR_ROF, 1.0)

    rolled_damage = base_damage * rolls[actual_damage_attr]
    rolled_rof = base_rof * rolls[ATTR_ROF]
    rolled_dps = rolled_damage / rolled_rof

    # Median DPS threshold (50th percentile)
    median_dps = np.median(rolled_dps)
    primary_success = rolled_dps >= median_dps

    # CPU catastrophe threshold (worst 10%)
    base_cpu = target.base_stats.get(ATTR_CPU, 1.0)
    rolled_cpu = base_cpu * rolls[ATTR_CPU]
    cpu_catastrophe_threshold = np.percentile(rolled_cpu, CATASTROPHE_PERCENTILE)
    secondary_ok = rolled_cpu < cpu_catastrophe_threshold

    combined_success = primary_success & secondary_ok

    return SuccessResult(
        success_rate=float(np.mean(combined_success)),
        p_primary=float(np.mean(primary_success)),
        p_secondary=float(np.mean(secondary_ok)),
        threshold=median_dps,
    )


def calc_dps_stat_based_success_rate(target: RollTarget, damage_attr: int = ATTR_DAMAGE) -> SuccessResult:
    """
    Calculate success rate for DPS modules using stat > base threshold.

    Used for pricing where we need sellable items (better than unrolled).
    """
    rolls = simulate_rolls(target)

    actual_damage_attr = damage_attr if damage_attr in rolls else ATTR_MISSILE_DAMAGE

    base_damage = target.base_stats.get(actual_damage_attr, 1.0)
    base_rof = target.base_stats.get(ATTR_ROF, 1.0)
    base_dps = base_damage / base_rof

    rolled_damage = base_damage * rolls[actual_damage_attr]
    rolled_rof = base_rof * rolls[ATTR_ROF]
    rolled_dps = rolled_damage / rolled_rof

    # Sellable = DPS > base DPS
    primary_success = rolled_dps > base_dps

    # CPU catastrophe
    base_cpu = target.base_stats.get(ATTR_CPU, 1.0)
    rolled_cpu = base_cpu * rolls[ATTR_CPU]
    cpu_catastrophe_threshold = np.percentile(rolled_cpu, CATASTROPHE_PERCENTILE)
    secondary_ok = rolled_cpu < cpu_catastrophe_threshold

    combined_success = primary_success & secondary_ok

    return SuccessResult(
        success_rate=float(np.mean(combined_success)),
        p_primary=float(np.mean(primary_success)),
        p_secondary=float(np.mean(secondary_ok)),
        threshold=base_dps,
        base_stat=base_dps,
    )


def calc_web_success_rate(target: RollTarget) -> SuccessResult:
    """Calculate success rate for stasis webifiers."""
    rolls = simulate_rolls(target)

    base_web = abs(target.base_stats.get(ATTR_VELOCITY_BONUS, -60))
    base_range = target.base_stats.get(ATTR_RANGE, 14000)

    rolled_web = base_web * rolls[ATTR_VELOCITY_BONUS]
    rolled_range = base_range * rolls[ATTR_RANGE]

    median_web = np.median(rolled_web)
    median_range = np.median(rolled_range)

    # Both web AND range must be >= median
    primary_success = (rolled_web >= median_web) & (rolled_range >= median_range)

    # CPU catastrophe
    base_cpu = target.base_stats.get(ATTR_CPU, 25)
    rolled_cpu = base_cpu * rolls[ATTR_CPU]
    cpu_catastrophe_threshold = np.percentile(rolled_cpu, CATASTROPHE_PERCENTILE)
    secondary_ok = rolled_cpu < cpu_catastrophe_threshold

    combined_success = primary_success & secondary_ok

    return SuccessResult(
        success_rate=float(np.mean(combined_success)),
        p_primary=float(np.mean(primary_success)),
        p_secondary=float(np.mean(secondary_ok)),
        threshold=f"Web>={median_web:.1f}%, Range>={median_range/1000:.1f}km",
    )


def calc_shield_booster_success_rate(target: RollTarget) -> SuccessResult:
    """Calculate success rate for shield boosters."""
    rolls = simulate_rolls(target)

    base_boost = target.base_stats.get(ATTR_SHIELD_BOOST, 256)
    base_duration = target.base_stats.get(ATTR_DURATION, 4000)

    rolled_boost = base_boost * rolls[ATTR_SHIELD_BOOST]
    rolled_duration = base_duration * rolls[ATTR_DURATION]
    rolled_boost_per_sec = rolled_boost / (rolled_duration / 1000)

    median_boost_per_sec = np.median(rolled_boost_per_sec)
    primary_success = rolled_boost_per_sec >= median_boost_per_sec

    # Cap catastrophe
    base_cap = target.base_stats.get(ATTR_CAP, 280)
    rolled_cap = base_cap * rolls[ATTR_CAP]
    cap_catastrophe_threshold = np.percentile(rolled_cap, CATASTROPHE_PERCENTILE)
    secondary_ok = rolled_cap < cap_catastrophe_threshold

    combined_success = primary_success & secondary_ok

    return SuccessResult(
        success_rate=float(np.mean(combined_success)),
        p_primary=float(np.mean(primary_success)),
        p_secondary=float(np.mean(secondary_ok)),
        threshold=median_boost_per_sec,
    )


def calc_plate_success_rate(target: RollTarget) -> SuccessResult:
    """Calculate success rate for armor plates."""
    rolls = simulate_rolls(target)

    base_hp = target.base_stats.get(ATTR_ARMOR_HP, 6000)
    rolled_hp = base_hp * rolls[ATTR_ARMOR_HP]

    median_hp = np.median(rolled_hp)
    primary_success = rolled_hp >= median_hp

    # Mass catastrophe
    base_mass = target.base_stats.get(ATTR_MASS, 2500000)
    rolled_mass = base_mass * rolls[ATTR_MASS]
    mass_catastrophe_threshold = np.percentile(rolled_mass, CATASTROPHE_PERCENTILE)
    secondary_ok = rolled_mass < mass_catastrophe_threshold

    combined_success = primary_success & secondary_ok

    return SuccessResult(
        success_rate=float(np.mean(combined_success)),
        p_primary=float(np.mean(primary_success)),
        p_secondary=float(np.mean(secondary_ok)),
        threshold=median_hp,
    )


def calc_ab_success_rate(target: RollTarget) -> SuccessResult:
    """Calculate success rate for afterburners."""
    rolls = simulate_rolls(target)

    base_velocity = target.base_stats.get(ATTR_VELOCITY_BONUS, 171)
    rolled_velocity = base_velocity * rolls[ATTR_VELOCITY_BONUS]

    median_velocity = np.median(rolled_velocity)
    primary_success = rolled_velocity >= median_velocity

    # Cap catastrophe
    base_cap = target.base_stats.get(ATTR_CAP, 135)
    rolled_cap = base_cap * rolls[ATTR_CAP]
    cap_catastrophe_threshold = np.percentile(rolled_cap, CATASTROPHE_PERCENTILE)
    secondary_ok = rolled_cap < cap_catastrophe_threshold

    combined_success = primary_success & secondary_ok

    return SuccessResult(
        success_rate=float(np.mean(combined_success)),
        p_primary=float(np.mean(primary_success)),
        p_secondary=float(np.mean(secondary_ok)),
        threshold=median_velocity,
    )


def calc_dda_success_rate(target: RollTarget) -> SuccessResult:
    """Calculate success rate for drone damage amplifiers."""
    rolls = simulate_rolls(target)

    base_damage = target.base_stats.get(ATTR_DDA_DAMAGE, 23.8)
    rolled_damage = base_damage * rolls[ATTR_DDA_DAMAGE]

    # Sellable = rolled damage > base damage
    primary_success = rolled_damage > base_damage

    # CPU catastrophe
    base_cpu = target.base_stats.get(ATTR_CPU, 45)
    rolled_cpu = base_cpu * rolls[ATTR_CPU]
    cpu_catastrophe_threshold = np.percentile(rolled_cpu, CATASTROPHE_PERCENTILE)
    secondary_ok = rolled_cpu < cpu_catastrophe_threshold

    combined_success = primary_success & secondary_ok

    return SuccessResult(
        success_rate=float(np.mean(combined_success)),
        p_primary=float(np.mean(primary_success)),
        p_secondary=float(np.mean(secondary_ok)),
        threshold=base_damage,
        base_stat=base_damage,
    )


def calc_shield_extender_success_rate(target: RollTarget) -> SuccessResult:
    """Calculate success rate for shield extenders."""
    rolls = simulate_rolls(target)

    base_hp = target.base_stats.get(ATTR_SHIELD_CAP, 2625)
    rolled_hp = base_hp * rolls[ATTR_SHIELD_CAP]

    # Sellable = rolled HP > base HP
    primary_success = rolled_hp > base_hp

    # CPU catastrophe
    base_cpu = target.base_stats.get(ATTR_CPU, 35)
    rolled_cpu = base_cpu * rolls[ATTR_CPU]
    cpu_catastrophe_threshold = np.percentile(rolled_cpu, CATASTROPHE_PERCENTILE)
    secondary_ok = rolled_cpu < cpu_catastrophe_threshold

    combined_success = primary_success & secondary_ok

    return SuccessResult(
        success_rate=float(np.mean(combined_success)),
        p_primary=float(np.mean(primary_success)),
        p_secondary=float(np.mean(secondary_ok)),
        threshold=base_hp,
        base_stat=base_hp,
    )


def calc_cap_battery_success_rate(target: RollTarget) -> SuccessResult:
    """Calculate success rate for cap batteries."""
    rolls = simulate_rolls(target)

    base_cap = target.base_stats.get(ATTR_CAP_CAPACITY, 1755)
    rolled_cap = base_cap * rolls[ATTR_CAP_CAPACITY]

    # Sellable = rolled cap > base cap
    primary_success = rolled_cap > base_cap

    # CPU catastrophe
    base_cpu = target.base_stats.get(ATTR_CPU, 42)
    rolled_cpu = base_cpu * rolls[ATTR_CPU]
    cpu_catastrophe_threshold = np.percentile(rolled_cpu, CATASTROPHE_PERCENTILE)
    secondary_ok = rolled_cpu < cpu_catastrophe_threshold

    combined_success = primary_success & secondary_ok

    return SuccessResult(
        success_rate=float(np.mean(combined_success)),
        p_primary=float(np.mean(primary_success)),
        p_secondary=float(np.mean(secondary_ok)),
        threshold=base_cap,
        base_stat=base_cap,
    )


def calc_warp_disruptor_success_rate(target: RollTarget) -> SuccessResult:
    """Calculate success rate for warp disruptors."""
    rolls = simulate_rolls(target)

    base_range = target.base_stats.get(ATTR_RANGE, 24000)
    rolled_range = base_range * rolls[ATTR_RANGE]

    # Sellable = range > base range
    primary_success = rolled_range > base_range

    # CPU catastrophe
    base_cpu = target.base_stats.get(ATTR_CPU, 30)
    rolled_cpu = base_cpu * rolls[ATTR_CPU]
    cpu_catastrophe_threshold = np.percentile(rolled_cpu, CATASTROPHE_PERCENTILE)
    secondary_ok = rolled_cpu < cpu_catastrophe_threshold

    combined_success = primary_success & secondary_ok

    return SuccessResult(
        success_rate=float(np.mean(combined_success)),
        p_primary=float(np.mean(primary_success)),
        p_secondary=float(np.mean(secondary_ok)),
        threshold=base_range,
        base_stat=base_range,
    )


def calc_damage_control_success_rate(target: RollTarget) -> SuccessResult:
    """Calculate success rate for damage controls."""
    rolls = simulate_rolls(target)

    # DC is special - only CPU matters, always sellable
    base_cpu = target.base_stats.get(ATTR_CPU, 30)
    rolled_cpu = base_cpu * rolls[ATTR_CPU]

    # Sellable = CPU < base CPU (any improvement)
    primary_success = rolled_cpu < base_cpu

    # No secondary catastrophe for DC
    secondary_ok = np.ones(len(rolled_cpu), dtype=bool)

    combined_success = primary_success & secondary_ok

    return SuccessResult(
        success_rate=float(np.mean(combined_success)),
        p_primary=float(np.mean(primary_success)),
        p_secondary=float(np.mean(secondary_ok)),
        threshold=base_cpu,
        base_stat=base_cpu,
    )


# Registry of module type -> calculator function
CALCULATORS: dict[str, Callable[[RollTarget], SuccessResult]] = {
    'dps': calc_dps_stat_based_success_rate,
    'bcs': lambda t: calc_dps_stat_based_success_rate(t, ATTR_MISSILE_DAMAGE),
    'web': calc_web_success_rate,
    'shield_booster': calc_shield_booster_success_rate,
    'armor_plate': calc_plate_success_rate,
    'afterburner': calc_ab_success_rate,
    'dda': calc_dda_success_rate,
    'shield_extender': calc_shield_extender_success_rate,
    'cap_battery': calc_cap_battery_success_rate,
    'warp_disruptor': calc_warp_disruptor_success_rate,
    'damage_control': calc_damage_control_success_rate,
}


def get_success_rate(target: RollTarget) -> SuccessResult:
    """
    Get success rate for any module type.

    Args:
        target: Roll target configuration

    Returns:
        SuccessResult with rates and threshold
    """
    calculator = CALCULATORS.get(target.module_type, calc_dps_stat_based_success_rate)
    return calculator(target)
