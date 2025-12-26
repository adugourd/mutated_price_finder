"""
Unit tests for Fuzzwork market data fetching.
"""

import pytest
import responses

from src.data.fuzzwork import get_jita_prices, get_sell_price, FUZZWORK_MARKET_URL, JITA_REGION


class TestGetJitaPrices:
    """Tests for Jita price fetching."""

    @responses.activate
    def test_fetches_single_item(self):
        """Successfully fetch price for a single item."""
        type_id = 41218
        mock_response = {
            str(type_id): {
                "sell": {"min": 28000000.0, "median": 30000000.0, "percentile": 29000000.0},
                "buy": {"max": 25000000.0}
            }
        }

        responses.add(
            responses.GET,
            f"{FUZZWORK_MARKET_URL}?region={JITA_REGION}&types={type_id}",
            json=mock_response,
            status=200
        )

        result = get_jita_prices([type_id])

        assert type_id in result
        assert result[type_id]['min'] == 28000000.0
        assert result[type_id]['median'] == 30000000.0

    @responses.activate
    def test_fetches_multiple_items(self):
        """Successfully fetch prices for multiple items."""
        type_ids = [41218, 49738]
        mock_response = {
            "41218": {"sell": {"min": 28000000.0, "median": 30000000.0, "percentile": 29000000.0}},
            "49738": {"sell": {"min": 55000000.0, "median": 60000000.0, "percentile": 57000000.0}},
        }

        responses.add(
            responses.GET,
            f"{FUZZWORK_MARKET_URL}?region={JITA_REGION}&types=41218,49738",
            json=mock_response,
            status=200
        )

        result = get_jita_prices(type_ids)

        assert len(result) == 2
        assert 41218 in result
        assert 49738 in result
        assert result[41218]['min'] == 28000000.0
        assert result[49738]['min'] == 55000000.0

    @responses.activate
    def test_handles_missing_item(self):
        """Handles items not found in market data."""
        type_id = 99999999
        mock_response = {}

        responses.add(
            responses.GET,
            f"{FUZZWORK_MARKET_URL}?region={JITA_REGION}&types={type_id}",
            json=mock_response,
            status=200
        )

        result = get_jita_prices([type_id])

        # Should return zeros for missing item
        assert type_id in result
        assert result[type_id]['min'] == 0

    def test_empty_type_list(self):
        """Handles empty type ID list without making API call."""
        result = get_jita_prices([])
        assert result == {}

    @responses.activate
    def test_handles_partial_data(self):
        """Handles items with partial price data."""
        type_id = 41218
        mock_response = {
            str(type_id): {
                "sell": {"min": 28000000.0}  # Missing median and percentile
            }
        }

        responses.add(
            responses.GET,
            f"{FUZZWORK_MARKET_URL}?region={JITA_REGION}&types={type_id}",
            json=mock_response,
            status=200
        )

        result = get_jita_prices([type_id])

        assert result[type_id]['min'] == 28000000.0
        assert result[type_id]['median'] == 0  # Default to 0
        assert result[type_id]['percentile'] == 0


class TestGetSellPrice:
    """Tests for single item price fetching."""

    @responses.activate
    def test_returns_min_price_by_default(self):
        """Returns minimum sell price by default."""
        type_id = 41218
        mock_response = {
            str(type_id): {
                "sell": {"min": 28000000.0, "median": 30000000.0}
            }
        }

        responses.add(
            responses.GET,
            f"{FUZZWORK_MARKET_URL}?region={JITA_REGION}&types={type_id}",
            json=mock_response,
            status=200
        )

        result = get_sell_price(type_id)

        assert result == 28000000.0

    @responses.activate
    def test_returns_specified_price_type(self):
        """Returns specified price type when requested."""
        type_id = 41218
        mock_response = {
            str(type_id): {
                "sell": {"min": 28000000.0, "median": 30000000.0}
            }
        }

        responses.add(
            responses.GET,
            f"{FUZZWORK_MARKET_URL}?region={JITA_REGION}&types={type_id}",
            json=mock_response,
            status=200
        )

        result = get_sell_price(type_id, price_type='median')

        assert result == 30000000.0

    @responses.activate
    def test_returns_zero_for_missing_item(self):
        """Returns zero for items not found."""
        type_id = 99999999
        mock_response = {}

        responses.add(
            responses.GET,
            f"{FUZZWORK_MARKET_URL}?region={JITA_REGION}&types={type_id}",
            json=mock_response,
            status=200
        )

        result = get_sell_price(type_id)

        assert result == 0
