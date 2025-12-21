"""
Integration test fixtures.

Provides mock HTTP responses and sample data for integration testing.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Any

import pytest
import numpy as np

# Fixtures directory
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def load_fixture(name: str) -> Any:
    """Load a JSON fixture file."""
    with open(FIXTURES_DIR / name) as f:
        return json.load(f)


@pytest.fixture
def mock_fuzzwork_response():
    """Mock Fuzzwork API response."""
    return load_fixture("fuzzwork_prices.json")


@pytest.fixture
def mock_fuzzwork_session(mock_fuzzwork_response):
    """Create a mock session that returns Fuzzwork price data."""
    mock_response = MagicMock()
    mock_response.json.return_value = mock_fuzzwork_response
    mock_response.status_code = 200
    mock_response.raise_for_status.return_value = None

    mock_session = MagicMock()
    mock_session.get.return_value = mock_response

    return mock_session


@pytest.fixture
def sample_contract_stats():
    """Sample contract stats for regression testing."""
    # Simulate DDA contract data
    # Stats represent damage bonus (e.g., 1.24 = 24% bonus)
    stats = np.array([
        1.20, 1.21, 1.22, 1.23, 1.24, 1.25, 1.26, 1.27, 1.28, 1.29,
        1.22, 1.23, 1.25, 1.26, 1.27, 1.24, 1.25, 1.26, 1.28, 1.27,
    ])

    # Prices in ISK (roughly correlate with stats)
    prices = np.array([
        500e6, 550e6, 600e6, 650e6, 700e6, 800e6, 900e6, 1000e6, 1100e6, 1200e6,
        580e6, 640e6, 780e6, 880e6, 980e6, 720e6, 810e6, 860e6, 1050e6, 970e6,
    ])

    return stats, prices


@pytest.fixture
def sample_dda_target():
    """Sample DDA roll target configuration."""
    from src.config.loader import RollTarget, MutaplasmidRange

    return RollTarget(
        key="test_dda_radical",
        name="Test DDA",
        base_type_id=33848,
        muta_type_id=56306,
        base_name="Dread Guristas DDA",
        muta_name="Radical DDA Muta",
        module_type="dda",
        base_stats={1255: 1.2515},  # Base DDA damage bonus
        muta_ranges=[
            MutaplasmidRange(attr_id=1255, min_mult=0.95, max_mult=1.145),  # Damage
            MutaplasmidRange(attr_id=50, min_mult=0.85, max_mult=1.3),  # CPU
        ],
        primary_desc="Damage bonus >= median",
        secondary_desc="CPU not catastrophic",
    )
