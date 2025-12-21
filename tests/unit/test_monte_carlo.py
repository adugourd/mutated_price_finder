"""
Unit tests for Monte Carlo simulation module.
"""

import numpy as np
import pytest

from src.analysis.monte_carlo import (
    simulate_rolls,
    get_success_rate,
    calc_dps_stat_based_success_rate,
    calc_dda_success_rate,
    SuccessResult,
)


class TestSimulateRolls:
    """Test roll simulation."""

    def test_returns_correct_attributes(self, sample_dps_target):
        """Simulated rolls contain all expected attribute IDs."""
        rolls = simulate_rolls(sample_dps_target, n_samples=1000)
        for muta_range in sample_dps_target.muta_ranges:
            assert muta_range.attr_id in rolls

    def test_roll_bounds(self, sample_dps_target):
        """All rolls fall within mutaplasmid bounds."""
        rolls = simulate_rolls(sample_dps_target, n_samples=10000)
        for muta_range in sample_dps_target.muta_ranges:
            roll_values = rolls[muta_range.attr_id]
            assert np.all(roll_values >= muta_range.min_mult)
            assert np.all(roll_values <= muta_range.max_mult)

    def test_uniform_distribution(self, sample_dps_target):
        """Rolls are approximately uniformly distributed."""
        rolls = simulate_rolls(sample_dps_target, n_samples=100000)
        for muta_range in sample_dps_target.muta_ranges:
            roll_values = rolls[muta_range.attr_id]
            midpoint = (muta_range.min_mult + muta_range.max_mult) / 2
            # Mean should be close to midpoint for uniform distribution
            assert abs(np.mean(roll_values) - midpoint) < 0.01

    def test_reproducibility_with_seed(self, sample_dps_target, seed_rng):
        """Setting random seed produces reproducible results."""
        np.random.seed(42)
        rolls1 = simulate_rolls(sample_dps_target, n_samples=100)
        np.random.seed(42)
        rolls2 = simulate_rolls(sample_dps_target, n_samples=100)
        for attr_id in rolls1:
            np.testing.assert_array_equal(rolls1[attr_id], rolls2[attr_id])


class TestSuccessRateCalculation:
    """Test success rate calculations."""

    def test_success_rate_bounds(self, sample_dps_target):
        """Success rate is between 0 and 1."""
        result = get_success_rate(sample_dps_target)
        assert 0 <= result.success_rate <= 1
        assert 0 <= result.p_primary <= 1
        assert 0 <= result.p_secondary <= 1

    def test_combined_less_than_individual(self, sample_dps_target):
        """Combined success rate <= min of individual rates."""
        result = get_success_rate(sample_dps_target)
        assert result.success_rate <= result.p_primary
        assert result.success_rate <= result.p_secondary

    def test_dps_success_rate_approximately_50_percent(self, sample_dps_target):
        """DPS modules should have ~50% primary success (median threshold)."""
        result = calc_dps_stat_based_success_rate(sample_dps_target)
        # Primary rate should be around 50% for "above median" criteria
        assert 0.45 <= result.p_primary <= 0.55

    def test_dda_success_rate_reasonable(self, sample_dda_target):
        """DDA success rate should be reasonable for its roll range."""
        result = calc_dda_success_rate(sample_dda_target)
        # With 0.8-1.2 range, ~50% should be above base
        assert 0.40 <= result.p_primary <= 0.55
        # Secondary (CPU not catastrophic) should be ~90%
        assert 0.85 <= result.p_secondary <= 0.95


class TestSuccessResult:
    """Test SuccessResult dataclass."""

    def test_success_result_attributes(self, sample_dps_target):
        """SuccessResult has all expected attributes."""
        result = get_success_rate(sample_dps_target)
        assert isinstance(result, SuccessResult)
        assert hasattr(result, 'success_rate')
        assert hasattr(result, 'p_primary')
        assert hasattr(result, 'p_secondary')
        assert hasattr(result, 'threshold')
