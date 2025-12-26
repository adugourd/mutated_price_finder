"""
Backtesting framework for price predictions.

Validates model performance using walk-forward validation on historical data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from src.models.training import MODULE_CONFIGS, ModuleTypeConfig


@dataclass
class BacktestResult:
    """Result of backtesting a prediction model."""
    module_type: str
    train_days: int
    test_days: int
    n_predictions: int
    mae: float  # Mean Absolute Error in ISK
    mape: float  # Mean Absolute Percentage Error
    r2_score: float
    directional_accuracy: float  # % of correct price direction predictions
    within_20pct: float  # % of predictions within 20% of actual
    within_50pct: float  # % of predictions within 50% of actual

    def __str__(self) -> str:
        """Format results for display."""
        return (
            f"Backtest: {self.module_type} ({self.train_days}d train / {self.test_days}d test)\n"
            f"  Predictions: {self.n_predictions}\n"
            f"  MAE:         {self.mae/1_000_000:.2f}M ISK\n"
            f"  MAPE:        {self.mape:.1f}%\n"
            f"  R² Score:    {self.r2_score:.3f}\n"
            f"  Direction:   {self.directional_accuracy:.1%}\n"
            f"  Within 20%:  {self.within_20pct:.1%}\n"
            f"  Within 50%:  {self.within_50pct:.1%}"
        )


def _gather_backtest_data(
    module_type: str,
    total_days: int,
    verbose: bool = True
) -> pd.DataFrame | None:
    """
    Gather historical data for backtesting.

    This is a simplified version that uses the training module's data gathering.
    """
    from src.models.training import gather_training_data, clean_training_data

    if module_type not in MODULE_CONFIGS:
        if verbose:
            print(f"Unknown module type: {module_type}")
        return None

    config = MODULE_CONFIGS[module_type]

    if verbose:
        print(f"Gathering {total_days} days of historical data for {module_type}...")

    try:
        raw_data = gather_training_data(config, total_days, verbose=verbose)
        if raw_data is None or raw_data.empty:
            return None

        # Add snapshot_date if not present (use current date as approximation)
        if 'snapshot_date' not in raw_data.columns:
            # Assign random dates within the range for testing
            dates = pd.date_range(
                end=datetime.now(),
                periods=len(raw_data),
                freq='h'
            )
            raw_data['snapshot_date'] = np.random.choice(dates, len(raw_data))

        return raw_data
    except Exception as e:
        if verbose:
            print(f"Error gathering data: {e}")
        return None


def _train_test_split_by_date(
    df: pd.DataFrame,
    test_days: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data by date for walk-forward validation."""
    if 'snapshot_date' not in df.columns:
        # Fall back to random split
        n_test = max(1, int(len(df) * 0.2))
        return df.iloc[:-n_test], df.iloc[-n_test:]

    df = df.copy()
    df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])

    cutoff_date = df['snapshot_date'].max() - timedelta(days=test_days)

    train = df[df['snapshot_date'] < cutoff_date]
    test = df[df['snapshot_date'] >= cutoff_date]

    return train, test


def backtest_model(
    module_type: str,
    train_days: int = 90,
    test_days: int = 30,
    verbose: bool = True,
) -> BacktestResult | None:
    """
    Backtest a price prediction model on historical data.

    Uses walk-forward validation: train on [T-train_days, T], predict on [T, T+test_days].

    Args:
        module_type: Module type to backtest (e.g., 'gyro', 'capbat')
        train_days: Days of training data
        test_days: Days of test data
        verbose: Print progress

    Returns:
        BacktestResult with performance metrics, or None if insufficient data
    """
    import xgboost as xgb
    from sklearn.metrics import mean_absolute_error, r2_score

    if module_type not in MODULE_CONFIGS:
        if verbose:
            print(f"Unknown module type: {module_type}")
        return None

    config = MODULE_CONFIGS[module_type]
    total_days = train_days + test_days + 10  # Buffer

    # Gather data
    data = _gather_backtest_data(module_type, total_days, verbose=verbose)
    if data is None or len(data) < 50:
        if verbose:
            print("Insufficient data for backtesting")
        return None

    # Split by date
    train_data, test_data = _train_test_split_by_date(data, test_days)

    if len(train_data) < 30 or len(test_data) < 10:
        if verbose:
            print(f"Insufficient split: {len(train_data)} train, {len(test_data)} test")
        return None

    if verbose:
        print(f"Train: {len(train_data)} contracts, Test: {len(test_data)} contracts")

    # Prepare features
    feature_names = config.feature_names

    # Verify all features exist
    missing = [f for f in feature_names if f not in train_data.columns]
    if missing:
        if verbose:
            print(f"Missing features: {missing}")
        return None

    # Train model
    X_train = train_data[feature_names].values
    y_train_raw = train_data['price'].values
    y_train = np.log1p(y_train_raw)  # Log transform

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Predict on test set
    X_test = test_data[feature_names].values
    y_test = test_data['price'].values

    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_pred = np.maximum(y_pred, 0)  # Ensure non-negative

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)

    # MAPE (avoid division by zero)
    valid_mask = y_test > 0
    if valid_mask.sum() > 0:
        mape = np.mean(np.abs((y_test[valid_mask] - y_pred[valid_mask]) / y_test[valid_mask])) * 100
    else:
        mape = float('inf')

    r2 = r2_score(y_test, y_pred)

    # Directional accuracy (above/below median)
    median_train = np.median(y_train_raw)
    actual_above_median = y_test > median_train
    pred_above_median = y_pred > median_train
    directional_accuracy = np.mean(actual_above_median == pred_above_median)

    # Within X% accuracy
    with np.errstate(divide='ignore', invalid='ignore'):
        pct_error = np.abs((y_pred - y_test) / y_test)
        pct_error = np.where(np.isfinite(pct_error), pct_error, 1.0)

    within_20pct = np.mean(pct_error < 0.20)
    within_50pct = np.mean(pct_error < 0.50)

    result = BacktestResult(
        module_type=module_type,
        train_days=train_days,
        test_days=test_days,
        n_predictions=len(y_test),
        mae=float(mae),
        mape=float(mape),
        r2_score=float(r2),
        directional_accuracy=float(directional_accuracy),
        within_20pct=float(within_20pct),
        within_50pct=float(within_50pct),
    )

    if verbose:
        print(f"\n{result}")

    return result


def run_all_backtests(
    train_days: int = 90,
    test_days: int = 30,
    verbose: bool = True
) -> list[BacktestResult]:
    """
    Run backtests for all supported module types.

    Args:
        train_days: Days of training data for each backtest
        test_days: Days of test data for each backtest
        verbose: Print progress

    Returns:
        List of BacktestResult objects
    """
    results = []

    for module_type in MODULE_CONFIGS:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Backtesting {module_type}")
            print('='*60)

        result = backtest_model(
            module_type,
            train_days=train_days,
            test_days=test_days,
            verbose=verbose
        )

        if result:
            results.append(result)

    if verbose and results:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print('='*60)
        print(f"{'Module':<12} {'MAE (M)':<10} {'MAPE':<8} {'R²':<8} {'Within 20%':<12}")
        print("-" * 60)
        for r in results:
            print(f"{r.module_type:<12} {r.mae/1e6:<10.2f} {r.mape:<8.1f}% {r.r2_score:<8.3f} {r.within_20pct:<12.1%}")

    return results


def print_backtest_summary(results: list[BacktestResult]) -> None:
    """Print a summary table of backtest results."""
    if not results:
        print("No backtest results to display.")
        return

    print(f"\n{'Module':<12} {'MAE (M)':<10} {'MAPE':<8} {'R²':<8} {'<20%':<8} {'<50%':<8}")
    print("-" * 60)

    for r in sorted(results, key=lambda x: x.within_20pct, reverse=True):
        print(
            f"{r.module_type:<12} "
            f"{r.mae/1e6:<10.2f} "
            f"{r.mape:<7.1f}% "
            f"{r.r2_score:<8.3f} "
            f"{r.within_20pct:<7.1%} "
            f"{r.within_50pct:<7.1%}"
        )


if __name__ == "__main__":
    """Run backtests when executed directly."""
    import argparse

    parser = argparse.ArgumentParser(description="Backtest price prediction models")
    parser.add_argument(
        "--module", "-m",
        help="Module type to backtest (default: all)",
        default=None
    )
    parser.add_argument(
        "--train-days",
        type=int,
        default=90,
        help="Days of training data (default: 90)"
    )
    parser.add_argument(
        "--test-days",
        type=int,
        default=30,
        help="Days of test data (default: 30)"
    )

    args = parser.parse_args()

    if args.module:
        result = backtest_model(
            args.module,
            train_days=args.train_days,
            test_days=args.test_days,
            verbose=True
        )
        if result:
            print(f"\nFinal result:\n{result}")
    else:
        results = run_all_backtests(
            train_days=args.train_days,
            test_days=args.test_days,
            verbose=True
        )
