"""
Integration tests for the data pipeline.

Tests the complete flow from data fetching to analysis.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest
import numpy as np

from src.analysis.regression import fit_constrained_regression
from src.analysis.monte_carlo import get_success_rate, simulate_rolls
from src.analysis.risk import calculate_roi, calculate_bankroll_risk
from src.formatters.isk import format_isk


@pytest.mark.integration
class TestPriceAnalysisPipeline:
    """Test the complete price analysis pipeline."""

    def test_regression_with_realistic_data(self, sample_contract_stats):
        """Test regression fits correctly with realistic contract data."""
        stats, prices = sample_contract_stats

        result = fit_constrained_regression(
            stats, prices,
            base_stat=1.18,  # Base DDA stat
            max_stat=1.30,   # Max possible with radical muta
        )

        assert result['method'] == 'constrained_regression'
        assert result['n_used'] > 10
        assert result['expected_price'] > 0
        assert result['slope'] > 0  # Price should increase with stat
        assert result['coverage_pct'] > 50  # Good data coverage

    def test_regression_handles_outliers(self, sample_contract_stats):
        """Test that regression correctly filters outliers."""
        stats, prices = sample_contract_stats

        # Add some outliers
        stats_with_outliers = np.append(stats, [1.25, 1.26])
        prices_with_outliers = np.append(prices, [50e9, 100e9])  # Extreme prices

        result = fit_constrained_regression(
            stats_with_outliers, prices_with_outliers,
            base_stat=1.18,
            max_stat=1.30,
        )

        assert result['n_outliers'] >= 1  # Should detect outliers
        assert result['expected_price'] < 50e9  # Should not be affected by outliers


@pytest.mark.integration
class TestMonteCarloIntegration:
    """Test Monte Carlo simulation integration."""

    def test_success_rate_calculation(self, sample_dda_target):
        """Test complete success rate calculation for a target."""
        result = get_success_rate(sample_dda_target)

        assert 0 <= result.success_rate <= 1
        assert 0 <= result.p_primary <= 1
        assert 0 <= result.p_secondary <= 1
        assert result.success_rate <= result.p_primary  # Combined <= individual
        assert result.success_rate <= result.p_secondary

    def test_simulate_rolls_produces_valid_distribution(self, sample_dda_target):
        """Test that simulated rolls follow expected distribution."""
        rolls = simulate_rolls(sample_dda_target, n_samples=10000)

        # Check damage attribute exists and is within range
        damage_attr_id = 1255
        assert damage_attr_id in rolls

        damage_rolls = rolls[damage_attr_id]
        muta_range = sample_dda_target.muta_ranges[0]

        # simulate_rolls returns multipliers, not final values
        min_expected = muta_range.min_mult
        max_expected = muta_range.max_mult

        assert np.min(damage_rolls) >= min_expected * 0.99  # Allow small tolerance
        assert np.max(damage_rolls) <= max_expected * 1.01

        # Distribution should be approximately uniform
        midpoint = (min_expected + max_expected) / 2
        below_mid = np.sum(damage_rolls < midpoint)
        above_mid = np.sum(damage_rolls >= midpoint)

        # Should be roughly 50/50
        ratio = below_mid / len(damage_rolls)
        assert 0.45 < ratio < 0.55


@pytest.mark.integration
class TestROIAnalysisIntegration:
    """Test ROI analysis integration."""

    def test_complete_roi_calculation(self, sample_dda_target):
        """Test complete ROI calculation workflow."""
        # Simulate what the main calculator does
        roll_cost = 88e6 + 352e6  # Base + muta
        success_result = get_success_rate(sample_dda_target)
        sale_price = 2.1e9  # Expected sale price

        roi = calculate_roi(roll_cost, success_result.success_rate, sale_price)

        assert 'expected_value' in roi
        assert 'expected_profit' in roi
        assert 'roi_pct' in roi
        assert 'rolls_per_success' in roi

        # With 45% success rate and 2.1B sale, should be profitable
        if success_result.success_rate > 0.2:
            assert roi['expected_value'] > 0

    def test_bankroll_risk_calculation(self, sample_dda_target):
        """Test bankroll risk analysis."""
        roll_cost = 440e6
        success_rate = 0.45
        sale_price = 2.1e9
        bankroll = 2e9  # 2 billion ISK

        risk = calculate_bankroll_risk(success_rate, roll_cost, sale_price, bankroll)

        assert 'n_rolls' in risk
        assert 'min_successes_needed' in risk
        assert 'prob_profitable' in risk
        assert 'expected_profit' in risk
        assert 'profit_at_5pct' in risk
        assert 'profit_at_50pct' in risk
        assert 'profit_at_95pct' in risk

        # With 2B bankroll, should be able to do ~4 rolls
        assert risk['n_rolls'] >= 4

        # Percentiles should be ordered
        assert risk['profit_at_5pct'] <= risk['profit_at_50pct']
        assert risk['profit_at_50pct'] <= risk['profit_at_95pct']


@pytest.mark.integration
class TestFormatterIntegration:
    """Test formatting integration."""

    def test_isk_formatting_end_to_end(self):
        """Test ISK formatting with various values from the pipeline."""
        test_cases = [
            (88_000_000, "88.0M"),
            (352_400_000, "352.4M"),
            (2_100_000_000, "2.10B"),
            (440_500_000, "440.5M"),
            (516_900_000, "516.9M"),
        ]

        for value, expected in test_cases:
            result = format_isk(value)
            assert result == expected, f"Expected {expected} for {value}, got {result}"
