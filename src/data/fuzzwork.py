"""
Fuzzwork Market API client for Jita prices.

Fetches market data from the Fuzzwork Market API for EVE Online items.
"""

from typing import Optional

import requests

from src.config.loader import load_constants

# Load API configuration
_constants = load_constants()
FUZZWORK_MARKET_URL = _constants['api']['fuzzwork_market_url']
JITA_REGION = _constants['api']['jita_region_id']


def get_jita_prices(type_ids: list[int]) -> dict[int, dict]:
    """
    Get Jita sell prices for multiple type IDs.

    Args:
        type_ids: List of EVE type IDs to fetch prices for

    Returns:
        Dict mapping type_id -> {'min': float, 'median': float, 'percentile': float}
    """
    if not type_ids:
        return {}

    type_str = ','.join(str(t) for t in type_ids)
    url = f"{FUZZWORK_MARKET_URL}?region={JITA_REGION}&types={type_str}"

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()

    prices = {}
    for type_id in type_ids:
        str_id = str(type_id)
        if str_id in data:
            sell_data = data[str_id].get('sell', {})
            prices[type_id] = {
                'min': float(sell_data.get('min', 0)),
                'median': float(sell_data.get('median', 0)),
                'percentile': float(sell_data.get('percentile', 0)),
            }
        else:
            prices[type_id] = {'min': 0, 'median': 0, 'percentile': 0}

    return prices


def get_sell_price(type_id: int, price_type: str = 'min') -> float:
    """
    Get Jita sell price for a single type ID.

    Args:
        type_id: EVE type ID
        price_type: Price metric to return ('min', 'median', or 'percentile')

    Returns:
        Price in ISK
    """
    prices = get_jita_prices([type_id])
    return prices.get(type_id, {}).get(price_type, 0)
