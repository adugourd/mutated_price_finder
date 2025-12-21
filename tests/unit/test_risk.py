"""
Unit tests for risk analysis module.
"""

import pytest
import numpy as np

from src.analysis.risk import (
    probability_of_profit,
    calculate_bankroll_risk,
    calculate_roi,
)


class TestProbabilityOfProfit:
    """Test binomial probability calculations."""

    def test_certain_success(self):
        """100% success rate means 100% profit probability."""
        prob = probability_of_profit(n_rolls=10, success_rate=1.0, min_successes=5)
        assert prob == 1.0

    def test_impossible_requirement(self):
        """Needing more successes than rolls means 0% probability."""
        prob = probability_of_profit(n_rolls=5, success_rate=0.5, min_successes=10)
        assert prob == 0.0

    def test_fair_coin_many_rolls(self):
        """50% success needing half is ~50% for large n."""
        prob = probability_of_profit(n_rolls=100, success_rate=0.5, min_successes=50)
        assert 0.45 < prob < 0.55

    def test_no_successes_needed(self):
        """If 0 successes needed, probability is 100%."""
        prob = probability_of_profit(n_rolls=10, success_rate=0.1, min_successes=0)
        assert prob == 1.0

    def test_edge_case_zero_rolls(self):
        """Zero rolls returns 0 probability."""
        prob = probability_of_profit(n_rolls=0, success_rate=0.5, min_successes=1)
        assert prob == 0.0


class TestBankrollRisk:
    """Test bankroll risk analysis."""

    def test_positive_ev_expected_profit(self):
        """Positive EV yields positive expected profit."""
        result = calculate_bankroll_risk(
            success_rate=0.5,
            roll_cost=100,
            expected_price=300,  # 50% * 300 = 150 > 100
            bankroll=1000
        )
        assert result['expected_profit'] > 0

    def test_negative_ev_expected_loss(self):
        """Negative EV yields negative expected profit."""
        result = calculate_bankroll_risk(
            success_rate=0.1,
            roll_cost=100,
            expected_price=200,  # 10% * 200 = 20 < 100
            bankroll=1000
        )
        assert result['expected_profit'] < 0

    def test_n_rolls_calculation(self):
        """Number of rolls is floor(bankroll / roll_cost)."""
        result = calculate_bankroll_risk(
            success_rate=0.5,
            roll_cost=100,
            expected_price=200,
            bankroll=550
        )
        assert result['n_rolls'] == 5  # floor(550/100)

    def test_percentile_ordering(self):
        """5th percentile <= median <= 95th percentile."""
        result = calculate_bankroll_risk(
            success_rate=0.5,
            roll_cost=100,
            expected_price=250,
            bankroll=10000
        )
        assert result['profit_at_5pct'] <= result['profit_at_50pct']
        assert result['profit_at_50pct'] <= result['profit_at_95pct']

    def test_zero_bankroll(self):
        """Zero bankroll returns zero rolls."""
        result = calculate_bankroll_risk(
            success_rate=0.5,
            roll_cost=100,
            expected_price=200,
            bankroll=0
        )
        assert result['n_rolls'] == 0

    def test_insufficient_bankroll(self):
        """Bankroll less than roll cost returns zero rolls."""
        result = calculate_bankroll_risk(
            success_rate=0.5,
            roll_cost=100,
            expected_price=200,
            bankroll=50
        )
        assert result['n_rolls'] == 0


class TestCalculateROI:
    """Test ROI calculation."""

    def test_positive_roi(self):
        """Positive expected value yields positive ROI."""
        result = calculate_roi(
            roll_cost=100,
            success_rate=0.5,
            sale_price=300
        )
        # EV = 0.5 * 300 = 150, profit = 50, ROI = 50%
        assert result['roi_pct'] == 50.0

    def test_negative_roi(self):
        """Negative expected value yields negative ROI."""
        result = calculate_roi(
            roll_cost=100,
            success_rate=0.2,
            sale_price=200
        )
        # EV = 0.2 * 200 = 40, profit = -60, ROI = -60%
        assert result['roi_pct'] == -60.0

    def test_breakeven_rate(self):
        """Breakeven rate is roll_cost / sale_price."""
        result = calculate_roi(
            roll_cost=100,
            success_rate=0.5,
            sale_price=200
        )
        assert result['breakeven_rate'] == 0.5  # 100/200

    def test_rolls_per_success(self):
        """Rolls per success is 1 / success_rate."""
        result = calculate_roi(
            roll_cost=100,
            success_rate=0.25,
            sale_price=500
        )
        assert result['rolls_per_success'] == 4.0  # 1/0.25

    def test_zero_cost_handling(self):
        """Zero roll cost returns zero ROI (edge case)."""
        result = calculate_roi(
            roll_cost=0,
            success_rate=0.5,
            sale_price=200
        )
        assert result['roi_pct'] == 0
