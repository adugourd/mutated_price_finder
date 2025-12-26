"""
Unit tests for XGBoost model training pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.models.training import (
    MODULE_CONFIGS,
    ModuleTypeConfig,
    TrainingResult,
)


class TestModuleConfigs:
    """Tests for module configuration."""

    def test_all_configs_have_required_fields(self):
        """All module configs have required fields."""
        for name, config in MODULE_CONFIGS.items():
            assert isinstance(config, ModuleTypeConfig), f"{name} is not ModuleTypeConfig"
            assert config.name, f"{name} missing name"
            assert len(config.source_type_ids) > 0, f"{name} has no source type IDs"
            assert len(config.feature_attrs) > 0, f"{name} has no feature attributes"
            assert len(config.feature_names) == len(config.feature_attrs), \
                f"{name} feature names/attrs mismatch"

    def test_feature_names_are_valid_identifiers(self):
        """Feature names are valid Python identifiers."""
        for name, config in MODULE_CONFIGS.items():
            for feature_name in config.feature_names:
                assert feature_name.isidentifier(), \
                    f"Invalid feature name '{feature_name}' in {name}"

    def test_expected_module_types_exist(self):
        """Expected module types are configured."""
        expected_types = ['capbat', 'gyro', 'heatsink', 'magstab', 'bcs', 'entropic', 'dda']
        for module_type in expected_types:
            assert module_type in MODULE_CONFIGS, f"Missing module type: {module_type}"

    def test_dda_has_correct_features(self):
        """DDA module has damage_bonus_pct and cpu features (no ROF)."""
        dda_config = MODULE_CONFIGS['dda']
        assert 'damage_bonus_pct' in dda_config.feature_names
        assert 'cpu' in dda_config.feature_names
        assert 'rof_multiplier' not in dda_config.feature_names

    def test_gyro_has_dps_features(self):
        """Gyro module has damage, rof, and cpu features."""
        gyro_config = MODULE_CONFIGS['gyro']
        assert 'damage_modifier' in gyro_config.feature_names
        assert 'rof_multiplier' in gyro_config.feature_names
        assert 'cpu' in gyro_config.feature_names


class TestTrainingResult:
    """Tests for TrainingResult dataclass."""

    def test_creates_valid_result(self):
        """Creates valid TrainingResult with all fields."""
        result = TrainingResult(
            model=None,  # Placeholder
            feature_names=['damage', 'rof', 'cpu'],
            r2_score=0.85,
            mae=50_000_000,
            n_contracts=500,
            trained_at=datetime.now(),
            days=30,
            module_type='gyro',
            feature_ranges={'damage': (1.0, 1.3), 'rof': (0.85, 0.95), 'cpu': (15, 30)},
        )

        assert result.r2_score == 0.85
        assert result.n_contracts == 500
        assert result.module_type == 'gyro'
        assert len(result.feature_names) == 3
        assert 'damage' in result.feature_ranges


class TestDataPreparation:
    """Tests for data preparation functions."""

    @pytest.fixture
    def sample_contract_data(self):
        """Create sample contract data for testing."""
        np.random.seed(42)
        n_samples = 100

        return pd.DataFrame({
            'contract_id': range(n_samples),
            'price': np.random.uniform(50_000_000, 500_000_000, n_samples),
            'damage_modifier': np.random.uniform(1.0, 1.3, n_samples),
            'rof_multiplier': np.random.uniform(0.85, 0.95, n_samples),
            'cpu': np.random.uniform(15, 30, n_samples),
            'source_type_id': np.random.choice([12058, 13937, 15681], n_samples),
        })

    def test_sample_data_has_expected_columns(self, sample_contract_data):
        """Sample data fixture has expected columns."""
        expected_cols = ['contract_id', 'price', 'damage_modifier', 'rof_multiplier', 'cpu']
        for col in expected_cols:
            assert col in sample_contract_data.columns

    def test_sample_data_has_reasonable_values(self, sample_contract_data):
        """Sample data has values in reasonable ranges."""
        assert sample_contract_data['price'].min() >= 50_000_000
        assert sample_contract_data['damage_modifier'].min() >= 1.0
        assert sample_contract_data['rof_multiplier'].max() <= 1.0


class TestFeatureRanges:
    """Tests for feature range tracking."""

    def test_feature_ranges_capture_min_max(self):
        """Feature ranges correctly capture min/max values."""
        df = pd.DataFrame({
            'damage': [1.05, 1.15, 1.25],
            'rof': [0.87, 0.90, 0.93],
            'cpu': [18, 22, 28],
        })

        feature_ranges = {
            name: (float(df[name].min()), float(df[name].max()))
            for name in ['damage', 'rof', 'cpu']
        }

        assert feature_ranges['damage'] == (1.05, 1.25)
        assert feature_ranges['rof'] == (0.87, 0.93)
        assert feature_ranges['cpu'] == (18, 28)

    def test_feature_ranges_handle_single_value(self):
        """Feature ranges handle single-value columns."""
        df = pd.DataFrame({
            'damage': [1.15, 1.15, 1.15],
        })

        feature_ranges = {
            'damage': (float(df['damage'].min()), float(df['damage'].max()))
        }

        assert feature_ranges['damage'] == (1.15, 1.15)


class TestPriceOutlierRemoval:
    """Tests for price outlier removal logic."""

    def test_iqr_removes_extreme_outliers(self):
        """IQR method removes extreme price outliers."""
        # Create data with obvious outliers
        prices = np.array([100, 110, 120, 130, 140, 150, 10000])

        q1 = np.percentile(prices, 25)
        q3 = np.percentile(prices, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        filtered = prices[(prices >= lower_bound) & (prices <= upper_bound)]

        # The extreme outlier (10000) should be removed
        assert 10000 not in filtered
        assert len(filtered) == 6

    def test_iqr_preserves_normal_distribution(self):
        """IQR method preserves most data in normal distribution."""
        np.random.seed(42)
        prices = np.random.normal(100_000_000, 20_000_000, 1000)
        prices = prices[prices > 0]  # Remove negatives

        q1 = np.percentile(prices, 25)
        q3 = np.percentile(prices, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        filtered = prices[(prices >= lower_bound) & (prices <= upper_bound)]

        # Should keep most data (~99.3% for normal distribution)
        retention_rate = len(filtered) / len(prices)
        assert retention_rate > 0.95
