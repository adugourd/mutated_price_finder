"""
XGBoost model training for mutated module price prediction.

Provides generic training logic that works for any module type
by extracting historical contract data and training XGBoost models.
"""

from __future__ import annotations

import re
import tarfile
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass

import requests
import pandas as pd
import numpy as np

# EVE Ref base URL for historical data
EVEREF_HISTORY_URL = "https://data.everef.net/public-contracts/history"

# Dogma Attribute IDs
ATTR_DAMAGE = 64
ATTR_ROF = 204
ATTR_CPU = 50
ATTR_POWER = 30
ATTR_CAP_CAPACITY = 67
ATTR_CAP_WARFARE_RESIST = 2267
ATTR_SHIELD_CAPACITY = 263


@dataclass
class ModuleTypeConfig:
    """Configuration for training a specific module type."""
    name: str
    source_type_ids: set[int]
    feature_attrs: list[int]  # Dogma attribute IDs to use as features
    feature_names: list[str]  # Human-readable names for features
    min_price: float = 1_000_000  # Minimum price filter (1M ISK)
    min_primary_stat: float | None = None  # Optional minimum for primary stat


# Module type configurations for XGBoost training
MODULE_CONFIGS: dict[str, ModuleTypeConfig] = {
    'capbat': ModuleTypeConfig(
        name='Large Cap Battery',
        source_type_ids={
            41218,  # Republic Fleet Large Cap Battery
            41220,  # Thukker Large Cap Battery
            3554,   # Large Cap Battery II
            3552,   # Large Cap Battery I
            41216,  # Dark Blood Large Cap Battery
            41214,  # True Sansha Large Cap Battery
        },
        feature_attrs=[ATTR_CAP_CAPACITY, ATTR_CPU, ATTR_POWER, ATTR_CAP_WARFARE_RESIST],
        feature_names=['cap_bonus', 'cpu', 'powergrid', 'cap_warfare_resist'],
        min_primary_stat=1500,  # Filter out non-large cap batteries
    ),
    'gyro': ModuleTypeConfig(
        name='Gyrostabilizer',
        source_type_ids={
            519,    # Gyrostabilizer II
            13939,  # Domination Gyrostabilizer
            15806,  # Republic Fleet Gyrostabilizer
        },
        feature_attrs=[ATTR_DAMAGE, ATTR_ROF, ATTR_CPU],
        feature_names=['damage_modifier', 'rof_multiplier', 'cpu'],
    ),
    'heatsink': ModuleTypeConfig(
        name='Heat Sink',
        source_type_ids={
            2364,   # Heat Sink II
            13943,  # Dark Blood Heat Sink
            13945,  # True Sansha Heat Sink
            15810,  # Imperial Navy Heat Sink
        },
        feature_attrs=[ATTR_DAMAGE, ATTR_ROF, ATTR_CPU],
        feature_names=['damage_modifier', 'rof_multiplier', 'cpu'],
    ),
    'magstab': ModuleTypeConfig(
        name='Magnetic Field Stabilizer',
        source_type_ids={
            10190,  # Magnetic Field Stabilizer II
            13947,  # Shadow Serpentis Mag Stab
            15895,  # Federation Navy Magnetic Field Stabilizer
        },
        feature_attrs=[ATTR_DAMAGE, ATTR_ROF, ATTR_CPU],
        feature_names=['damage_modifier', 'rof_multiplier', 'cpu'],
    ),
    'bcs': ModuleTypeConfig(
        name='Ballistic Control System',
        source_type_ids={
            22291,  # Ballistic Control System II
            13935,  # Domination BCS
            13937,  # Republic Fleet BCS
            15681,  # Caldari Navy BCS
        },
        feature_attrs=[ATTR_DAMAGE, ATTR_ROF, ATTR_CPU],
        feature_names=['damage_modifier', 'rof_multiplier', 'cpu'],
    ),
    'entropic': ModuleTypeConfig(
        name='Entropic Radiation Sink',
        source_type_ids={
            47911,  # Entropic Radiation Sink II
            48419,  # Veles Entropic Radiation Sink
            48421,  # Mystic Entropic Radiation Sink
        },
        feature_attrs=[ATTR_DAMAGE, ATTR_ROF, ATTR_CPU],
        feature_names=['damage_modifier', 'rof_multiplier', 'cpu'],
    ),
}


def download_snapshot(date: str, cache_dir: Path) -> Path | None:
    """Download a single daily snapshot (first one of the day)."""
    year = date[:4]
    cache_path = cache_dir / f"snapshot_{date}.tar.bz2"

    if cache_path.exists():
        return cache_path

    # Try to find the first snapshot of the day
    day_url = f"{EVEREF_HISTORY_URL}/{year}/{date}/"

    try:
        response = requests.get(day_url, timeout=10)
        response.raise_for_status()

        # Parse the HTML to find the first file
        content = response.text
        matches = re.findall(
            rf'href="[^"]*/(public-contracts-{date}_\d{{2}}-\d{{2}}-\d{{2}}\.v2\.tar\.bz2)"',
            content
        )

        if not matches:
            return None

        # Download the first snapshot
        file_name = matches[0]
        file_url = f"{day_url}{file_name}"

        file_response = requests.get(file_url, stream=True, timeout=120)
        file_response.raise_for_status()

        with open(cache_path, 'wb') as f:
            for chunk in file_response.iter_content(chunk_size=8192):
                f.write(chunk)

        return cache_path

    except Exception:
        return None


def extract_module_data(archive_path: Path, config: ModuleTypeConfig) -> pd.DataFrame:
    """Extract contract data for a specific module type from a snapshot archive."""
    try:
        with tarfile.open(archive_path, 'r:bz2') as tar:
            contracts_df = None
            dynamic_items_df = None
            dogma_attrs_df = None

            for member in tar.getmembers():
                if member.name.endswith('contracts.csv'):
                    f = tar.extractfile(member)
                    if f:
                        contracts_df = pd.read_csv(f)
                elif member.name.endswith('contract_dynamic_items.csv'):
                    f = tar.extractfile(member)
                    if f:
                        dynamic_items_df = pd.read_csv(f)
                elif member.name.endswith('contract_dynamic_items_dogma_attributes.csv'):
                    f = tar.extractfile(member)
                    if f:
                        dogma_attrs_df = pd.read_csv(f)

            if contracts_df is None or dynamic_items_df is None or dogma_attrs_df is None:
                return pd.DataFrame()

            # Filter for item exchange contracts
            contracts_df = contracts_df[contracts_df['type'] == 'item_exchange']

            # Filter for module source types
            dynamic_items_df = dynamic_items_df[
                dynamic_items_df['source_type_id'].isin(config.source_type_ids)
            ]

            if dynamic_items_df.empty:
                return pd.DataFrame()

            # Get dogma attributes for these items
            item_ids = dynamic_items_df['item_id'].unique()
            dogma_attrs_df = dogma_attrs_df[dogma_attrs_df['item_id'].isin(item_ids)]

            # Pivot each feature attribute
            result = dynamic_items_df.copy()
            for attr_id, attr_name in zip(config.feature_attrs, config.feature_names):
                attr_values = dogma_attrs_df[dogma_attrs_df['attribute_id'] == attr_id][['item_id', 'value']]
                attr_values = attr_values.rename(columns={'value': attr_name})
                result = result.merge(attr_values, on='item_id', how='left')

            # Merge with contracts for price
            result = result.merge(
                contracts_df[['contract_id', 'price']],
                on='contract_id',
                how='inner'
            )

            return result

    except Exception:
        return pd.DataFrame()


def gather_historical_data(
    module_type: str,
    days: int = 90,
    cache_dir: Path | None = None,
    verbose: bool = True
) -> pd.DataFrame:
    """Gather historical contract data for a module type."""
    if module_type not in MODULE_CONFIGS:
        raise ValueError(f"Unknown module type: {module_type}. Available: {list(MODULE_CONFIGS.keys())}")

    config = MODULE_CONFIGS[module_type]

    if cache_dir is None:
        cache_dir = Path(tempfile.gettempdir()) / "everef_history"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Generate date list
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    if verbose:
        print(f"Gathering {config.name} data for {len(dates)} days")

    all_data = []

    for i, date in enumerate(dates):
        if verbose and (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(dates)}] Processing {date}...")

        archive_path = download_snapshot(date, cache_dir)
        if archive_path is None:
            continue

        df = extract_module_data(archive_path, config)
        if not df.empty:
            df['snapshot_date'] = date
            all_data.append(df)

    if not all_data:
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)

    if verbose:
        print(f"  Total records before deduplication: {len(combined)}")

    return combined


def deduplicate_contracts(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate contracts that appear across multiple snapshots."""
    if df.empty:
        return df

    deduped = df.groupby('contract_id').first().reset_index()
    return deduped


def clean_data(
    df: pd.DataFrame,
    config: ModuleTypeConfig,
    verbose: bool = True
) -> pd.DataFrame:
    """Clean and filter the data for modeling."""
    if df.empty:
        return df

    df = df.copy()

    # Remove free transfers and obviously mispriced items
    df = df[df['price'] > config.min_price]

    # Apply primary stat filter if configured
    if config.min_primary_stat is not None and len(config.feature_names) > 0:
        primary_col = config.feature_names[0]
        if primary_col in df.columns:
            df = df[df[primary_col] > config.min_primary_stat]

    # Remove rows with missing values
    df = df.dropna(subset=config.feature_names + ['price'])

    # Remove extreme outliers using IQR on price
    Q1 = df['price'].quantile(0.25)
    Q3 = df['price'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers_removed = len(df[(df['price'] < lower) | (df['price'] > upper)])
    df = df[(df['price'] >= lower) & (df['price'] <= upper)]

    if verbose:
        print(f"  After cleaning: {len(df)} contracts (removed {outliers_removed} price outliers)")

    return df


@dataclass
class TrainingResult:
    """Result of training an XGBoost model."""
    model: object  # XGBRegressor
    feature_names: list[str]
    r2_score: float
    mae: float
    n_contracts: int
    trained_at: datetime
    days: int
    module_type: str
    feature_ranges: dict[str, tuple[float, float]] | None = None  # {name: (min, max)}


def train_xgboost(
    df: pd.DataFrame,
    feature_names: list[str],
    module_type: str,
    days: int,
    verbose: bool = True
) -> TrainingResult:
    """Train XGBoost model to predict price from stats."""
    try:
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, r2_score
    except ImportError:
        import subprocess
        subprocess.check_call(['pip', 'install', 'xgboost', 'scikit-learn'])
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, r2_score

    X = df[feature_names].values
    y = df['price'].values

    # Log transform price for better modeling
    y_log = np.log1p(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )

    if verbose:
        print("  Training XGBoost model...")

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Evaluate
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_test_actual = np.expm1(y_test)

    mae = mean_absolute_error(y_test_actual, y_pred)
    r2 = r2_score(y_test_actual, y_pred)

    if verbose:
        print(f"  R² Score: {r2:.3f}")
        print(f"  MAE: {mae/1_000_000:.2f}M ISK")

    # Calculate observed feature ranges from training data
    feature_ranges = {}
    for name in feature_names:
        feature_ranges[name] = (float(df[name].min()), float(df[name].max()))

    if verbose:
        print(f"  Feature ranges (observed in contracts):")
        for name, (min_val, max_val) in feature_ranges.items():
            print(f"    {name}: [{min_val:.2f}, {max_val:.2f}]")

    return TrainingResult(
        model=model,
        feature_names=feature_names,
        r2_score=r2,
        mae=mae,
        n_contracts=len(df),
        trained_at=datetime.now(),
        days=days,
        module_type=module_type,
        feature_ranges=feature_ranges,
    )


def train_model(
    module_type: str,
    days: int = 90,
    cache_dir: Path | None = None,
    verbose: bool = True
) -> TrainingResult | None:
    """
    Full training pipeline for a module type.

    Args:
        module_type: Type of module (e.g., 'capbat', 'gyro')
        days: Number of days of historical data
        cache_dir: Directory for caching downloaded data
        verbose: Whether to print progress

    Returns:
        TrainingResult with the trained model, or None if training failed
    """
    if module_type not in MODULE_CONFIGS:
        raise ValueError(f"Unknown module type: {module_type}")

    config = MODULE_CONFIGS[module_type]

    if verbose:
        print(f"Training {config.name} model with {days} days of data...")

    # Gather data
    raw_data = gather_historical_data(module_type, days, cache_dir, verbose)

    if raw_data.empty:
        if verbose:
            print("  No data gathered!")
        return None

    # Deduplicate
    unique_data = deduplicate_contracts(raw_data)

    if verbose:
        print(f"  After deduplication: {len(unique_data)} unique contracts")

    # Clean
    clean = clean_data(unique_data, config, verbose)

    if clean.empty or len(clean) < 10:
        if verbose:
            print("  Insufficient data after cleaning!")
        return None

    # Train
    return train_xgboost(clean, config.feature_names, module_type, days, verbose)


def predict_price(model: object, features: dict[str, float], feature_names: list[str]) -> float:
    """
    Predict price for a single item.

    Args:
        model: Trained XGBRegressor
        features: Dict mapping feature names to values
        feature_names: List of feature names in model order

    Returns:
        Predicted price in ISK
    """
    X = np.array([[features[name] for name in feature_names]])
    y_pred_log = model.predict(X)
    return float(np.expm1(y_pred_log)[0])
