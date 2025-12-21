"""
Bankroll risk analysis using binomial distribution.

Calculates probability of profit and risk metrics for mutaplasmid rolling.
"""

import numpy as np
from scipy import stats


def probability_of_profit(n_rolls: int, success_rate: float, min_successes: int) -> float:
    """
    Calculate probability of achieving at least min_successes in n_rolls.

    Uses binomial distribution CDF.

    Args:
        n_rolls: Total number of rolls
        success_rate: Probability of success per roll (0-1)
        min_successes: Minimum number of successes needed

    Returns:
        Probability of profit (0-1)
    """
    if n_rolls <= 0 or success_rate <= 0:
        return 0.0
    if min_successes > n_rolls:
        return 0.0
    if success_rate >= 1:
        return 1.0

    # P(X >= min_successes) = 1 - P(X < min_successes) = 1 - P(X <= min_successes - 1)
    return 1 - stats.binom.cdf(min_successes - 1, n_rolls, success_rate)


def calculate_bankroll_risk(
    success_rate: float,
    roll_cost: float,
    expected_price: float,
    bankroll: float,
) -> dict:
    """
    Calculate bankroll risk metrics.

    Given a bankroll, calculates how many rolls you can afford and
    the probability distribution of outcomes.

    Args:
        success_rate: Probability of sellable roll (0-1)
        roll_cost: Cost per roll (base + mutaplasmid)
        expected_price: Expected sale price for successful rolls
        bankroll: Available ISK

    Returns:
        Dict with risk metrics:
        - n_rolls: Number of rolls possible
        - min_successes_needed: Minimum successes to break even
        - breakeven_k: Exact breakeven point (fractional)
        - prob_profitable: Probability of being profitable
        - expected_profit: Expected total profit
        - profit_at_5pct: 5th percentile profit (worst reasonable case)
        - profit_at_50pct: Median profit
        - profit_at_95pct: 95th percentile profit (best reasonable case)
    """
    if roll_cost <= 0 or bankroll <= 0:
        return {
            'n_rolls': 0,
            'min_successes_needed': 0,
            'breakeven_k': 0,
            'prob_profitable': 0,
            'expected_profit': 0,
            'profit_at_5pct': 0,
            'profit_at_50pct': 0,
            'profit_at_95pct': 0,
        }

    # Number of rolls we can afford
    n_rolls = int(bankroll // roll_cost)

    if n_rolls == 0:
        return {
            'n_rolls': 0,
            'min_successes_needed': 0,
            'breakeven_k': 0,
            'prob_profitable': 0,
            'expected_profit': -bankroll,
            'profit_at_5pct': -bankroll,
            'profit_at_50pct': -bankroll,
            'profit_at_95pct': -bankroll,
        }

    # Total cost for all rolls
    total_cost = n_rolls * roll_cost

    # Breakeven: k successes where k * expected_price >= total_cost
    if expected_price > 0:
        breakeven_k = total_cost / expected_price
        min_successes_needed = int(np.ceil(breakeven_k))
    else:
        breakeven_k = float('inf')
        min_successes_needed = n_rolls + 1

    # Probability of profit
    prob_profitable = probability_of_profit(n_rolls, success_rate, min_successes_needed)

    # Expected profit = E[successes] * sale_price - total_cost
    expected_successes = n_rolls * success_rate
    expected_profit = expected_successes * expected_price - total_cost

    # Percentile calculations using binomial distribution
    def profit_at_k(k):
        return k * expected_price - total_cost

    # Find k values for percentiles
    # 5th percentile (worst reasonable case)
    k_5pct = stats.binom.ppf(0.05, n_rolls, success_rate)
    profit_at_5pct = profit_at_k(k_5pct)

    # 50th percentile (median)
    k_50pct = stats.binom.ppf(0.50, n_rolls, success_rate)
    profit_at_50pct = profit_at_k(k_50pct)

    # 95th percentile (best reasonable case)
    k_95pct = stats.binom.ppf(0.95, n_rolls, success_rate)
    profit_at_95pct = profit_at_k(k_95pct)

    return {
        'n_rolls': n_rolls,
        'min_successes_needed': min_successes_needed,
        'breakeven_k': breakeven_k,
        'prob_profitable': prob_profitable,
        'expected_profit': expected_profit,
        'profit_at_5pct': profit_at_5pct,
        'profit_at_50pct': profit_at_50pct,
        'profit_at_95pct': profit_at_95pct,
    }


def calculate_roi(roll_cost: float, success_rate: float, sale_price: float) -> dict:
    """
    Calculate ROI metrics for a single roll.

    Args:
        roll_cost: Cost per roll (base + mutaplasmid)
        success_rate: Probability of sellable roll (0-1)
        sale_price: Expected sale price for successful rolls

    Returns:
        Dict with ROI metrics
    """
    if roll_cost <= 0:
        return {
            'roll_cost': 0,
            'expected_value': 0,
            'expected_profit': 0,
            'roi_pct': 0,
            'breakeven_rate': 1,
            'rolls_per_success': float('inf'),
        }

    # Expected value per roll = success_rate * sale_price
    expected_value = success_rate * sale_price

    # Expected profit = expected_value - roll_cost
    expected_profit = expected_value - roll_cost

    # ROI = expected_profit / roll_cost
    roi_pct = (expected_profit / roll_cost) * 100

    # Break-even success rate
    breakeven_rate = roll_cost / sale_price if sale_price > 0 else 1

    # Rolls per success on average
    rolls_per_success = 1 / success_rate if success_rate > 0 else float('inf')

    return {
        'roll_cost': roll_cost,
        'expected_value': expected_value,
        'expected_profit': expected_profit,
        'roi_pct': roi_pct,
        'breakeven_rate': breakeven_rate,
        'rolls_per_success': rolls_per_success,
    }
