"""
Pytest configuration and shared fixtures.
"""

import pytest
import numpy as np

from src.config.loader import RollTarget, MutaplasmidRange, load_attributes

# Load attribute IDs
_attrs = load_attributes()


@pytest.fixture
def sample_dps_target():
    """Sample DPS module target (Heat Sink) for testing."""
    return RollTarget(
        key="test_heatsink",
        name="Test Heat Sink",
        base_type_id=15810,
        base_name="Imperial Navy Heat Sink",
        muta_type_id=49729,
        muta_name="Unstable Heat Sink Mutaplasmid",
        base_stats={
            _attrs['damage']: 1.12,
            _attrs['rof']: 0.89,
            _attrs['cpu']: 20
        },
        muta_ranges=[
            MutaplasmidRange(attr_id=_attrs['damage'], min_mult=0.98, max_mult=1.02, high_is_good=True),
            MutaplasmidRange(attr_id=_attrs['rof'], min_mult=0.975, max_mult=1.025, high_is_good=True),
            MutaplasmidRange(attr_id=_attrs['cpu'], min_mult=0.8, max_mult=1.5, high_is_good=False),
        ],
        module_type="dps",
        primary_desc="DPS > base",
        secondary_desc="CPU not in worst 10%",
    )


@pytest.fixture
def sample_dda_target():
    """Sample DDA target for testing."""
    return RollTarget(
        key="test_dda",
        name="Test DDA",
        base_type_id=33846,
        base_name="Dread Guristas Drone Damage Amplifier",
        muta_type_id=60476,
        muta_name="Radical Drone Damage Amplifier Mutaplasmid",
        base_stats={
            _attrs['dda_damage']: 23.8,
            _attrs['cpu']: 20
        },
        muta_ranges=[
            MutaplasmidRange(attr_id=_attrs['dda_damage'], min_mult=0.8, max_mult=1.2, high_is_good=True),
            MutaplasmidRange(attr_id=_attrs['cpu'], min_mult=0.7, max_mult=1.5, high_is_good=False),
        ],
        module_type="dda",
        primary_desc="Drone damage > base",
        secondary_desc="CPU not in worst 10%",
    )


@pytest.fixture
def sample_linear_data():
    """Perfect linear data for regression testing."""
    stats = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
    prices = np.array([100, 150, 200, 250, 300])  # y = 100*x
    return stats, prices


@pytest.fixture
def sample_noisy_data():
    """Noisy data with one outlier for regression testing."""
    stats = np.array([1.1, 1.2, 1.3, 1.4, 1.5, 1.6])
    prices = np.array([110, 120, 130, 140, 150, 5000])  # Last is outlier
    return stats, prices


@pytest.fixture
def seed_rng():
    """Seed random number generator for reproducible tests."""
    np.random.seed(42)
    yield
    # Reset to random state after test
    np.random.seed(None)
