"""
Unit tests for regression module.
"""

import numpy as np
import pytest

from src.analysis.regression import (
    fit_constrained_regression,
    remove_outliers_iqr,
    estimate_price_at_stat,
)


class TestRemoveOutliersIQR:
    """Test IQR-based outlier removal."""

    def test_no_outliers(self):
        """Normal data has no outliers removed."""
        values = np.array([100, 110, 120, 130, 140])
        filtered, n_removed = remove_outliers_iqr(values)
        assert n_removed == 0
        assert len(filtered) == 5

    def test_removes_extreme_outliers(self):
        """Extreme values are removed."""
        values = np.array([100, 110, 120, 130, 10000])
        filtered, n_removed = remove_outliers_iqr(values)
        assert n_removed == 1
        assert 10000 not in filtered

    def test_small_array_unchanged(self):
        """Arrays with < 4 elements are unchanged."""
        values = np.array([100, 10000])
        filtered, n_removed = remove_outliers_iqr(values)
        assert n_removed == 0
        assert len(filtered) == 2


class TestConstrainedRegression:
    """Test constrained linear regression."""

    def test_perfect_linear_data(self, sample_linear_data):
        """Regression handles perfectly linear data."""
        stats, prices = sample_linear_data
        result = fit_constrained_regression(stats, prices, base_stat=0.9, max_stat=3.5)
        # Slope should be close to 100 (y = 100x)
        assert result['method'] == 'constrained_regression'
        assert result['n_used'] > 0

    def test_filters_below_base(self):
        """Items at or below base stat are excluded (sellable = stat > base)."""
        stats = np.array([0.8, 0.9, 1.0, 1.1, 1.2])  # Three at or below base
        prices = np.array([50, 60, 100, 110, 120])
        result = fit_constrained_regression(stats, prices, base_stat=1.0, max_stat=1.5)
        assert result['n_sellable'] == 2  # Only 1.1, 1.2 (1.0 is not > 1.0)

    def test_coverage_calculation(self):
        """Coverage percentage is correctly computed."""
        stats = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
        prices = np.array([110, 120, 130, 140, 150])
        result = fit_constrained_regression(stats, prices, base_stat=1.0, max_stat=2.0)
        # Data covers [1.1, 1.5] of [1.0, 2.0] range = ~40%
        assert 30 < result['coverage_pct'] < 50

    def test_outlier_removal(self, sample_noisy_data):
        """IQR outliers are removed."""
        stats, prices = sample_noisy_data
        result = fit_constrained_regression(stats, prices, base_stat=1.0, max_stat=2.0)
        assert result['n_outliers'] >= 1

    def test_insufficient_data_fallback(self):
        """Falls back gracefully with insufficient data."""
        stats = np.array([1.1])
        prices = np.array([100])
        result = fit_constrained_regression(stats, prices, base_stat=1.0, max_stat=2.0)
        assert result['method'] in ('simple_mean', 'insufficient_data')

    def test_expected_price_non_negative(self):
        """Expected price is never negative."""
        stats = np.array([1.1, 1.2, 1.3])
        prices = np.array([100, 90, 80])  # Decreasing prices
        result = fit_constrained_regression(stats, prices, base_stat=1.0, max_stat=2.0)
        assert result['expected_price'] >= 0


class TestEstimatePriceAtStat:
    """Test price estimation from regression parameters."""

    def test_at_anchor_point(self):
        """Price at anchor equals anchor price."""
        price = estimate_price_at_stat(
            slope=100,
            anchor_stat=1.0,
            anchor_price=100,
            target_stat=1.0
        )
        assert price == 100

    def test_linear_extrapolation(self):
        """Price increases linearly with stat."""
        price = estimate_price_at_stat(
            slope=100,
            anchor_stat=1.0,
            anchor_price=100,
            target_stat=2.0
        )
        assert price == 200  # 100 + 100 * (2.0 - 1.0)

    def test_non_negative(self):
        """Price is never negative even with negative slope."""
        price = estimate_price_at_stat(
            slope=-1000,
            anchor_stat=1.0,
            anchor_price=100,
            target_stat=2.0
        )
        assert price == 0  # Would be -900, clamped to 0
