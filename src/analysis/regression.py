"""
Price regression models for mutated module valuation.

Fits constrained linear regression models to contract price data
for estimating expected sale prices.
"""

from __future__ import annotations

import numpy as np

from src.config.loader import load_constants

# Load thresholds from config
_constants = load_constants()
IQR_MULTIPLIER = _constants['analysis']['iqr_multiplier']


def remove_outliers_iqr(values: np.ndarray, multiplier: float | None = None) -> tuple[np.ndarray, int]:
    """
    Remove outliers using IQR method.

    Args:
        values: Array of values to filter
        multiplier: IQR multiplier for outlier bounds (default from config)

    Returns:
        Tuple of (filtered_values, n_outliers_removed)
    """
    if multiplier is None:
        multiplier = IQR_MULTIPLIER

    if len(values) < 4:
        return values, 0

    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1

    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    mask = (values >= lower_bound) & (values <= upper_bound)
    n_outliers = len(values) - np.sum(mask)

    return values[mask], n_outliers


def fit_constrained_regression(
    stats: np.ndarray,
    prices: np.ndarray,
    base_stat: float,
    max_stat: float,
) -> dict:
    """
    Fit linear regression constrained through the lowest data point.

    The regression is anchored at the worst sellable item to prevent
    overestimation of lower-stat item prices.

    Args:
        stats: Array of stat values for each item
        prices: Array of prices for each item
        base_stat: Base item stat value (sellable threshold)
        max_stat: Maximum possible stat value

    Returns:
        Dict with regression results including expected price at midpoint
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
            'anchor_stat': None,
            'coverage_pct': 0,
            'data_min_stat': None,
            'data_max_stat': None,
            'method': 'insufficient_data',
        }

    # Step 1: Filter to sellable items (stat > base_stat)
    sellable_mask = stats > base_stat
    sellable_stats = stats[sellable_mask]
    sellable_prices = prices[sellable_mask]
    n_sellable = len(sellable_stats)

    full_range = max_stat - base_stat

    if n_sellable < 3:
        # Calculate coverage for limited data
        if n_sellable > 0 and full_range > 0:
            data_min = np.min(sellable_stats)
            data_max = np.max(sellable_stats)
            min_pct = (data_min - base_stat) / full_range * 100
            max_pct = (data_max - base_stat) / full_range * 100
            coverage = max_pct - min_pct
        else:
            coverage = 0
            data_min = None
            data_max = None

        return {
            'n_total': len(stats),
            'n_sellable': n_sellable,
            'n_outliers': 0,
            'n_used': n_sellable,
            'expected_price': np.mean(sellable_prices) if n_sellable > 0 else 0,
            'midpoint_stat': (base_stat + max_stat) / 2,
            'slope': 0,
            'anchor_price': np.min(sellable_prices) if n_sellable > 0 else 0,
            'anchor_stat': data_min,
            'coverage_pct': coverage,
            'data_min_stat': data_min,
            'data_max_stat': data_max,
            'method': 'simple_mean',
        }

    # Step 2: Remove price outliers using IQR
    # Sort by stat to pair correctly
    sort_idx = np.argsort(sellable_stats)
    sorted_stats = sellable_stats[sort_idx]
    sorted_prices = sellable_prices[sort_idx]

    # IQR on prices
    q1 = np.percentile(sorted_prices, 25)
    q3 = np.percentile(sorted_prices, 75)
    iqr = q3 - q1

    if iqr > 0:
        lower_bound = q1 - IQR_MULTIPLIER * iqr
        upper_bound = q3 + IQR_MULTIPLIER * iqr
        outlier_mask = (sorted_prices >= lower_bound) & (sorted_prices <= upper_bound)
    else:
        outlier_mask = np.ones(len(sorted_prices), dtype=bool)

    filtered_stats = sorted_stats[outlier_mask]
    filtered_prices = sorted_prices[outlier_mask]
    n_outliers = len(sorted_prices) - len(filtered_prices)

    if len(filtered_stats) < 2:
        mean_price = np.mean(sellable_prices)
        return {
            'n_total': len(stats),
            'n_sellable': n_sellable,
            'n_outliers': n_outliers,
            'n_used': len(filtered_stats),
            'expected_price': mean_price,
            'midpoint_stat': (base_stat + max_stat) / 2,
            'slope': 0,
            'anchor_price': mean_price,
            'anchor_stat': None,
            'coverage_pct': 0,
            'data_min_stat': None,
            'data_max_stat': None,
            'method': 'mean_after_outlier_removal',
        }

    # Step 3: Find anchor point (lowest stat item)
    anchor_idx = np.argmin(filtered_stats)
    anchor_stat = filtered_stats[anchor_idx]
    anchor_price = filtered_prices[anchor_idx]

    # Step 4: Fit regression constrained through anchor
    # y - anchor_price = slope * (x - anchor_stat)
    # slope = sum((x - anchor_stat) * (y - anchor_price)) / sum((x - anchor_stat)^2)
    x_centered = filtered_stats - anchor_stat
    y_centered = filtered_prices - anchor_price

    denominator = np.sum(x_centered ** 2)
    if denominator > 0:
        slope = np.sum(x_centered * y_centered) / denominator
    else:
        slope = 0

    # Step 5: Calculate expected price at midpoint
    midpoint_stat = (base_stat + max_stat) / 2
    expected_price = anchor_price + slope * (midpoint_stat - anchor_stat)

    # Ensure non-negative
    expected_price = max(0, expected_price)

    # Calculate data coverage
    data_min_stat = np.min(filtered_stats)
    data_max_stat = np.max(filtered_stats)

    if full_range > 0:
        min_pct = (data_min_stat - base_stat) / full_range * 100
        max_pct = (data_max_stat - base_stat) / full_range * 100
        coverage_pct = max_pct - min_pct
    else:
        coverage_pct = 0

    return {
        'n_total': len(stats),
        'n_sellable': n_sellable,
        'n_outliers': n_outliers,
        'n_used': len(filtered_stats),
        'expected_price': expected_price,
        'midpoint_stat': midpoint_stat,
        'slope': slope,
        'anchor_price': anchor_price,
        'anchor_stat': anchor_stat,
        'coverage_pct': coverage_pct,
        'data_min_stat': data_min_stat,
        'data_max_stat': data_max_stat,
        'method': 'constrained_regression',
    }


def estimate_price_at_stat(
    slope: float,
    anchor_stat: float,
    anchor_price: float,
    target_stat: float,
) -> float:
    """
    Estimate price at a given stat value using regression parameters.

    Args:
        slope: Regression slope (price per unit stat)
        anchor_stat: Anchor point stat value
        anchor_price: Anchor point price
        target_stat: Stat value to estimate price for

    Returns:
        Estimated price
    """
    return max(0, anchor_price + slope * (target_stat - anchor_stat))
