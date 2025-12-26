"""
Shared utilities for CLI tools.

Provides common functionality used by find_prices.py, roi_calculator.py, and other CLI scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config.loader import RollTarget, load_all_targets, load_constants
from src.data.fuzzwork import get_jita_prices
from src.data.everef import download_contract_archive, extract_csv_from_archive
from src.models import get_or_train_model, ModelInfo, MODULE_CONFIGS


def get_market_prices(type_ids: list[int]) -> dict[int, dict]:
    """
    Fetch Jita market prices for given type IDs.

    Args:
        type_ids: List of EVE type IDs

    Returns:
        Dict mapping type_id -> {'min': sell_price, 'median': median_price, ...}
    """
    return get_jita_prices(type_ids)


def get_roll_cost(target: RollTarget, prices: dict[int, dict]) -> float:
    """
    Calculate total roll cost for a target.

    Args:
        target: Roll target configuration
        prices: Market prices dict from get_market_prices()

    Returns:
        Total cost (base item + mutaplasmid) in ISK
    """
    base_price = prices.get(target.base_type_id, {}).get('min', 0)
    muta_price = prices.get(target.muta_type_id, {}).get('min', 0)
    return base_price + muta_price


def load_contract_data(archive_path: Path | None = None) -> dict[str, pd.DataFrame]:
    """
    Download (if needed) and extract contract data.

    Args:
        archive_path: Optional path to existing archive, downloads if None

    Returns:
        Dict with 'contracts', 'items', 'dynamic_items', 'dogma_attributes' DataFrames
    """
    if archive_path is None:
        archive_path = download_contract_archive()

    data = {}

    data['contracts'] = extract_csv_from_archive(archive_path, 'contracts.csv')
    data['items'] = extract_csv_from_archive(archive_path, 'contract_items.csv')
    data['dynamic_items'] = extract_csv_from_archive(archive_path, 'contract_dynamic_items.csv')
    data['dogma_attributes'] = extract_csv_from_archive(
        archive_path, 'contract_dynamic_items_dogma_attributes.csv'
    )

    return data


# Mapping from target module_type to XGBoost model type
TARGET_TO_XGBOOST: dict[str, str] = {
    'cap_battery': 'capbat',
    'gyro': 'gyro',
    'heatsink': 'heatsink',
    'magstab': 'magstab',
    'bcs': 'bcs',
    'entropic': 'entropic',
    'dda': 'dda',
    'dps': 'gyro',  # Default DPS to gyro
}


def get_model_for_target(
    target: RollTarget,
    training_days: int | None = None,
    max_age_days: int | None = None,
    verbose: bool = True,
) -> ModelInfo | None:
    """
    Get XGBoost model for a target's module type.

    Args:
        target: Roll target configuration
        training_days: Days of training data (uses config default if None)
        max_age_days: Max model age before retraining (uses config default if None)
        verbose: Print progress

    Returns:
        ModelInfo or None if module type not supported
    """
    xgb_type = TARGET_TO_XGBOOST.get(target.module_type)
    if xgb_type is None:
        return None

    if xgb_type not in MODULE_CONFIGS:
        return None

    # Get defaults from config
    try:
        constants = load_constants()
        if training_days is None:
            training_days = constants.get('models', {}).get('find_prices_training_days', 180)
        if max_age_days is None:
            max_age_days = constants.get('models', {}).get('max_model_age_days', 10)
    except Exception:
        if training_days is None:
            training_days = 180
        if max_age_days is None:
            max_age_days = 10

    return get_or_train_model(
        xgb_type,
        training_days,
        max_age_days=max_age_days,
        verbose=verbose,
    )


def collect_target_type_ids(targets: dict[str, RollTarget] | None = None) -> set[int]:
    """
    Collect all type IDs needed for price lookup from targets.

    Args:
        targets: Dict of roll targets (loads all if None)

    Returns:
        Set of all base and mutaplasmid type IDs
    """
    if targets is None:
        targets = load_all_targets()

    type_ids = set()
    for target in targets.values():
        type_ids.add(target.base_type_id)
        type_ids.add(target.muta_type_id)
    return type_ids


def filter_item_exchange_contracts(contracts: pd.DataFrame) -> pd.DataFrame:
    """
    Filter contracts to only item exchange type.

    Args:
        contracts: Full contracts DataFrame

    Returns:
        Filtered DataFrame with only item_exchange contracts
    """
    return contracts[contracts['type'] == 'item_exchange'].copy()


def filter_single_item_contracts(
    dynamic_items: pd.DataFrame,
    contract_ids: pd.Index | None = None,
) -> pd.DataFrame:
    """
    Filter to contracts with exactly one dynamic item (no bundles).

    Args:
        dynamic_items: Dynamic items DataFrame
        contract_ids: Optional filter to specific contracts

    Returns:
        Filtered dynamic items DataFrame
    """
    if contract_ids is not None:
        dynamic_items = dynamic_items[dynamic_items['contract_id'].isin(contract_ids)]

    # Count items per contract
    items_per_contract = dynamic_items.groupby('contract_id').size()
    single_item_contracts = items_per_contract[items_per_contract == 1].index

    return dynamic_items[dynamic_items['contract_id'].isin(single_item_contracts)].copy()


def get_attribute_values(
    dogma_attrs: pd.DataFrame,
    item_ids: pd.Series | list,
    attribute_id: int,
    column_name: str = 'value',
) -> pd.DataFrame:
    """
    Extract specific attribute values for items.

    Args:
        dogma_attrs: Dogma attributes DataFrame
        item_ids: Item IDs to filter
        attribute_id: EVE dogma attribute ID
        column_name: Name for the value column

    Returns:
        DataFrame with 'item_id' and the value column
    """
    result = dogma_attrs[
        (dogma_attrs['item_id'].isin(item_ids)) &
        (dogma_attrs['attribute_id'] == attribute_id)
    ][['item_id', 'value']].copy()

    return result.rename(columns={'value': column_name})
