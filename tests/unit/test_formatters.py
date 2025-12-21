"""
Unit tests for formatting modules.
"""

import pytest

from src.formatters.isk import format_isk, format_isk_full
from src.formatters.stats import format_stat, format_stat_range


class TestFormatISK:
    """Test ISK value formatting."""

    @pytest.mark.parametrize("value,expected", [
        (0, "0"),
        (999, "999"),
        (1000, "1.0K"),
        (1500, "1.5K"),
        (1_000_000, "1.0M"),
        (1_500_000, "1.5M"),
        (1_000_000_000, "1.00B"),
        (2_500_000_000, "2.50B"),
    ])
    def test_positive_values(self, value, expected):
        """Positive values are formatted correctly."""
        assert format_isk(value) == expected

    @pytest.mark.parametrize("value,expected", [
        (-1000, "-1.0K"),
        (-1_000_000, "-1.0M"),
        (-1_000_000_000, "-1.00B"),
    ])
    def test_negative_values(self, value, expected):
        """Negative values include minus sign."""
        assert format_isk(value) == expected


class TestFormatISKFull:
    """Test full ISK formatting with commas."""

    def test_with_commas(self):
        """Values include thousand separators."""
        assert format_isk_full(1_500_000_000) == "1,500,000,000"
        assert format_isk_full(1234567) == "1,234,567"


class TestFormatStat:
    """Test stat value formatting by module type."""

    def test_dps_format(self):
        """DPS modules use multiplier format."""
        assert format_stat(1.234, 'dps') == "1.234x"

    def test_dda_format(self):
        """DDA modules use percentage format."""
        assert format_stat(25.5, 'dda') == "25.50%"

    def test_shield_extender_format(self):
        """Shield extenders use HP format."""
        assert format_stat(3000, 'shield_extender') == "3000 HP"

    def test_warp_disruptor_format(self):
        """Warp disruptors use km format."""
        assert format_stat(30000, 'warp_disruptor') == "30.0km"


class TestFormatStatRange:
    """Test stat range formatting."""

    def test_dps_range(self):
        """DPS range is formatted correctly."""
        result = format_stat_range(1.12, 1.14, 'dps')
        assert result == "[1.120x -> 1.140x]"

    def test_dda_range(self):
        """DDA range is formatted correctly."""
        result = format_stat_range(23.8, 28.6, 'dda')
        assert result == "[23.80% -> 28.60%]"
