"""
Analysis modules for mutated module valuation.
"""

from src.analysis.monte_carlo import (
    simulate_rolls,
    get_success_rate,
    SuccessResult,
    CALCULATORS,
)
from src.analysis.regression import (
    fit_constrained_regression,
    remove_outliers_iqr,
    estimate_price_at_stat,
)
from src.analysis.risk import (
    calculate_bankroll_risk,
    calculate_roi,
    probability_of_profit,
)

__all__ = [
    # Monte Carlo
    'simulate_rolls',
    'get_success_rate',
    'SuccessResult',
    'CALCULATORS',
    # Regression
    'fit_constrained_regression',
    'remove_outliers_iqr',
    'estimate_price_at_stat',
    # Risk
    'calculate_bankroll_risk',
    'calculate_roi',
    'probability_of_profit',
]
