"""
XGBoost model lifecycle management.

Handles loading, saving, and staleness checking for trained XGBoost models.
Models are stored with metadata for automatic retraining when stale.

Uses joblib for model serialization (more robust than pickle for sklearn/xgboost).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Any

import joblib

from src.config.loader import load_constants, PROJECT_ROOT
from src.models.training import (
    train_model,
    TrainingResult,
    MODULE_CONFIGS,
    predict_price as _predict_price,
)


@dataclass
class ModelInfo:
    """Information about a trained model."""
    model: Any  # XGBRegressor
    feature_names: list[str]
    r2_score: float
    mae: float
    n_contracts: int
    trained_at: datetime
    days: int
    module_type: str
    feature_ranges: dict[str, tuple[float, float]] | None = None  # {name: (min, max)}

    @property
    def age_days(self) -> float:
        """Age of the model in days."""
        return (datetime.now() - self.trained_at).total_seconds() / 86400

    def is_stale(self, max_age_days: int = 10) -> bool:
        """Check if the model is older than max_age_days."""
        return self.age_days > max_age_days


def _get_model_dir() -> Path:
    """Get the models directory, creating it if needed."""
    try:
        constants = load_constants()
        model_dir_name = constants.get('models', {}).get('model_dir', 'models')
    except Exception:
        model_dir_name = 'models'

    model_dir = PROJECT_ROOT / model_dir_name
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def _get_model_path(module_type: str, days: int) -> Path:
    """Get the path for a model file (joblib format)."""
    return _get_model_dir() / f"{module_type}_{days}d.joblib"


def _get_legacy_model_path(module_type: str, days: int) -> Path:
    """Get the path for a legacy pickle model file."""
    return _get_model_dir() / f"{module_type}_{days}d.pkl"


def save_model(result: TrainingResult) -> Path:
    """
    Save a trained model to disk using joblib.

    Args:
        result: TrainingResult from training

    Returns:
        Path to the saved model file
    """
    model_path = _get_model_path(result.module_type, result.days)

    data = {
        'model': result.model,
        'feature_names': result.feature_names,
        'r2_score': result.r2_score,
        'mae': result.mae,
        'n_contracts': result.n_contracts,
        'trained_at': result.trained_at,
        'days': result.days,
        'module_type': result.module_type,
        'feature_ranges': result.feature_ranges,
    }

    # Use joblib with compression for better performance
    joblib.dump(data, model_path, compress=3)

    # Remove legacy pickle file if it exists
    legacy_path = _get_legacy_model_path(result.module_type, result.days)
    if legacy_path.exists():
        legacy_path.unlink()

    return model_path


def load_model(module_type: str, days: int) -> ModelInfo | None:
    """
    Load a trained model from disk.

    Supports both joblib (new) and pickle (legacy) formats.
    Legacy pickle files are automatically migrated to joblib on load.

    Args:
        module_type: Type of module (e.g., 'capbat', 'gyro')
        days: Training data duration in days

    Returns:
        ModelInfo if model exists, None otherwise
    """
    model_path = _get_model_path(module_type, days)
    legacy_path = _get_legacy_model_path(module_type, days)

    data = None
    needs_migration = False

    # Try loading joblib format first
    if model_path.exists():
        try:
            data = joblib.load(model_path)
        except Exception:
            pass

    # Fall back to legacy pickle format
    if data is None and legacy_path.exists():
        try:
            with open(legacy_path, 'rb') as f:
                data = pickle.load(f)
            needs_migration = True
        except Exception:
            pass

    if data is None:
        return None

    try:
        model_info = ModelInfo(
            model=data['model'],
            feature_names=data['feature_names'],
            r2_score=data['r2_score'],
            mae=data['mae'],
            n_contracts=data['n_contracts'],
            trained_at=data['trained_at'],
            days=data['days'],
            module_type=data['module_type'],
            feature_ranges=data.get('feature_ranges'),  # Backwards compatible
        )

        # Migrate legacy pickle to joblib
        if needs_migration:
            joblib.dump(data, model_path, compress=3)
            legacy_path.unlink()

        return model_info
    except Exception:
        return None


def model_exists(module_type: str, days: int) -> bool:
    """Check if a model file exists (joblib or legacy pickle)."""
    return _get_model_path(module_type, days).exists() or _get_legacy_model_path(module_type, days).exists()


def get_model_age(module_type: str, days: int) -> float | None:
    """
    Get the age of a model in days.

    Returns:
        Age in days, or None if model doesn't exist
    """
    info = load_model(module_type, days)
    if info is None:
        return None
    return info.age_days


def get_or_train_model(
    module_type: str,
    days: int,
    max_age_days: int = 10,
    force_retrain: bool = False,
    verbose: bool = True
) -> ModelInfo | None:
    """
    Get a model, training it if necessary.

    This is the main entry point for getting a model. It will:
    1. Load existing model if valid and not stale
    2. Retrain if model is stale or doesn't exist
    3. Save the newly trained model

    Args:
        module_type: Type of module (e.g., 'capbat', 'gyro')
        days: Training data duration in days
        max_age_days: Maximum age before retraining
        force_retrain: Force retraining even if model is valid
        verbose: Whether to print progress

    Returns:
        ModelInfo with the model, or None if training failed
    """
    if module_type not in MODULE_CONFIGS:
        if verbose:
            print(f"Unknown module type: {module_type}")
        return None

    # Try to load existing model
    if not force_retrain:
        info = load_model(module_type, days)

        if info is not None:
            if not info.is_stale(max_age_days):
                if verbose:
                    print(f"Using cached {module_type} model (age: {info.age_days:.1f} days, R²: {info.r2_score:.3f})")
                return info
            else:
                if verbose:
                    print(f"Model is stale ({info.age_days:.1f} days old), retraining...")

    # Train new model
    if verbose:
        print(f"Training {module_type} model with {days} days of data...")

    result = train_model(module_type, days, verbose=verbose)

    if result is None:
        if verbose:
            print(f"Failed to train {module_type} model")
        return None

    # Save the model
    model_path = save_model(result)

    if verbose:
        print(f"Model saved to {model_path}")

    return ModelInfo(
        model=result.model,
        feature_names=result.feature_names,
        r2_score=result.r2_score,
        mae=result.mae,
        n_contracts=result.n_contracts,
        trained_at=result.trained_at,
        days=result.days,
        module_type=result.module_type,
        feature_ranges=result.feature_ranges,
    )


def predict_price(model_info: ModelInfo, features: dict[str, float]) -> float:
    """
    Predict price for a single item using a loaded model.

    Args:
        model_info: ModelInfo with the trained model
        features: Dict mapping feature names to values

    Returns:
        Predicted price in ISK
    """
    return _predict_price(model_info.model, features, model_info.feature_names)


def list_available_models() -> list[dict]:
    """
    List all available trained models.

    Returns:
        List of dicts with model info
    """
    model_dir = _get_model_dir()
    models = []
    seen = set()  # Track seen (module_type, days) to avoid duplicates

    # Check both joblib and legacy pickle files
    for pattern in ("*.joblib", "*.pkl"):
        for model_path in model_dir.glob(pattern):
            try:
                # Parse filename: {module_type}_{days}d.{ext}
                name = model_path.stem
                parts = name.rsplit('_', 1)
                if len(parts) != 2:
                    continue

                module_type = parts[0]
                days_str = parts[1].rstrip('d')
                days = int(days_str)

                # Skip if we've already seen this model
                key = (module_type, days)
                if key in seen:
                    continue
                seen.add(key)

                info = load_model(module_type, days)
                if info:
                    models.append({
                        'module_type': module_type,
                        'days': days,
                        'age_days': info.age_days,
                        'r2_score': info.r2_score,
                        'n_contracts': info.n_contracts,
                        'path': str(_get_model_path(module_type, days)),
                    })
            except Exception:
                continue

    return models


def delete_model(module_type: str, days: int) -> bool:
    """
    Delete a trained model (both joblib and legacy pickle formats).

    Returns:
        True if model was deleted, False if it didn't exist
    """
    deleted = False
    model_path = _get_model_path(module_type, days)
    legacy_path = _get_legacy_model_path(module_type, days)

    if model_path.exists():
        model_path.unlink()
        deleted = True
    if legacy_path.exists():
        legacy_path.unlink()
        deleted = True

    return deleted


def get_feature_importance(model_info: ModelInfo) -> dict[str, float]:
    """
    Get feature importance scores from the model.

    Uses the model's built-in feature importance (gain-based for XGBoost).

    Args:
        model_info: Trained model

    Returns:
        Dict mapping feature names to importance scores (normalized to sum to 1)
    """
    importance = model_info.model.feature_importances_
    total = sum(importance)

    if total == 0:
        return {name: 0.0 for name in model_info.feature_names}

    return {
        name: float(imp / total)
        for name, imp in zip(model_info.feature_names, importance)
    }


def print_feature_importance(model_info: ModelInfo) -> None:
    """
    Print formatted feature importance analysis.

    Args:
        model_info: Trained model with feature importances
    """
    importance = get_feature_importance(model_info)

    print("\nFeature Importance:")
    print("-" * 45)

    # Sort by importance (highest first)
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    for name, imp in sorted_features:
        bar_length = int(imp * 30)
        bar = "█" * bar_length
        print(f"  {name:20s} {imp:6.1%} {bar}")


@dataclass
class PredictionWithConfidence:
    """Prediction result with confidence interval."""
    prediction: float  # Point prediction (median/mean)
    lower: float  # Lower bound of confidence interval
    upper: float  # Upper bound of confidence interval
    confidence_level: float  # e.g., 0.80 for 80% CI

    def __str__(self) -> str:
        from src.formatters.isk import format_isk
        return (
            f"{format_isk(self.prediction)} "
            f"[{format_isk(self.lower)} - {format_isk(self.upper)}] "
            f"({self.confidence_level:.0%} CI)"
        )


def predict_with_confidence(
    model_info: ModelInfo,
    features: dict[str, float],
    confidence_level: float = 0.80,
) -> PredictionWithConfidence:
    """
    Predict price with confidence interval.

    Uses the model's MAE to estimate prediction uncertainty.
    The confidence interval is based on the historical prediction error.

    Args:
        model_info: Trained model
        features: Feature values for prediction
        confidence_level: Confidence level (0.0-1.0), default 0.80 for 80% CI

    Returns:
        PredictionWithConfidence with point estimate and bounds
    """
    import numpy as np

    # Get point prediction
    point_pred = predict_price(model_info, features)

    # Estimate uncertainty based on MAE
    # For a rough confidence interval, we use:
    # - 80% CI: ~1.28 standard deviations
    # - 90% CI: ~1.645 standard deviations
    # - 95% CI: ~1.96 standard deviations
    # We approximate std from MAE as: std ≈ MAE * 1.25 (for normal distribution)

    z_scores = {
        0.80: 1.28,
        0.90: 1.645,
        0.95: 1.96,
        0.99: 2.576,
    }
    z = z_scores.get(confidence_level, 1.28)

    # Estimate standard deviation from MAE
    estimated_std = model_info.mae * 1.25

    # Calculate bounds (use percentage-based for prices which are always positive)
    # This creates a multiplicative interval that works better for price data
    relative_error = estimated_std / max(point_pred, 1)
    multiplier = np.exp(z * relative_error)

    lower = point_pred / multiplier
    upper = point_pred * multiplier

    return PredictionWithConfidence(
        prediction=point_pred,
        lower=lower,
        upper=upper,
        confidence_level=confidence_level,
    )


def predict_range_with_confidence(
    model_info: ModelInfo,
    n_samples: int = 10000,
    confidence_level: float = 0.80,
) -> dict:
    """
    Predict price range using Monte Carlo simulation with confidence bounds.

    Samples from the observed feature ranges and predicts prices,
    then computes percentiles for the price distribution.

    Args:
        model_info: Trained model with feature_ranges
        n_samples: Number of Monte Carlo samples
        confidence_level: Confidence level for the interval

    Returns:
        Dict with 'median', 'lower', 'upper', 'mean', 'std' keys
    """
    import numpy as np

    if model_info.feature_ranges is None:
        return None

    # Generate random samples within feature ranges
    samples = {}
    for name in model_info.feature_names:
        if name not in model_info.feature_ranges:
            return None
        min_val, max_val = model_info.feature_ranges[name]
        samples[name] = np.random.uniform(min_val, max_val, n_samples)

    # Build feature matrix
    X = np.column_stack([samples[name] for name in model_info.feature_names])

    # Predict
    y_pred_log = model_info.model.predict(X)
    prices = np.expm1(y_pred_log)
    prices = np.maximum(prices, 0)

    # Calculate percentiles
    alpha = (1 - confidence_level) / 2
    lower_pct = alpha * 100
    upper_pct = (1 - alpha) * 100

    return {
        'median': float(np.median(prices)),
        'mean': float(np.mean(prices)),
        'std': float(np.std(prices)),
        'lower': float(np.percentile(prices, lower_pct)),
        'upper': float(np.percentile(prices, upper_pct)),
        'min': float(np.min(prices)),
        'max': float(np.max(prices)),
        'confidence_level': confidence_level,
    }


def get_feature_importance_summary(model_info: ModelInfo) -> str:
    """
    Get a single-line summary of feature importance.

    Args:
        model_info: Trained model

    Returns:
        String like "damage_modifier: 45%, rof_multiplier: 35%, cpu: 20%"
    """
    importance = get_feature_importance(model_info)
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    parts = [f"{name}: {imp:.0%}" for name, imp in sorted_features]
    return ", ".join(parts)


@dataclass
class MonteCarloResult:
    """Result of Monte Carlo price simulation."""
    mean_price: float
    median_price: float
    std_price: float
    percentile_5: float
    percentile_25: float
    percentile_75: float
    percentile_95: float
    n_samples: int
    prices: Any  # numpy array of all predicted prices


def simulate_prices_monte_carlo(
    model_info: ModelInfo,
    n_samples: int = 100_000,
) -> MonteCarloResult | None:
    """
    Simulate roll outcomes and predict prices using XGBoost.

    Uses the observed feature ranges from training data to constrain
    the simulation to realistic values (avoiding out-of-distribution predictions).

    Args:
        model_info: Trained model with feature_ranges
        n_samples: Number of Monte Carlo samples

    Returns:
        MonteCarloResult with price distribution statistics
    """
    import numpy as np

    if model_info.feature_ranges is None:
        return None

    # Generate random samples within observed feature ranges
    samples = {}
    for name in model_info.feature_names:
        if name not in model_info.feature_ranges:
            return None
        min_val, max_val = model_info.feature_ranges[name]
        samples[name] = np.random.uniform(min_val, max_val, n_samples)

    # Build feature matrix
    X = np.column_stack([samples[name] for name in model_info.feature_names])

    # Predict prices (model uses log-transformed prices)
    y_pred_log = model_info.model.predict(X)
    prices = np.expm1(y_pred_log)

    # Ensure no negative prices
    prices = np.maximum(prices, 0)

    return MonteCarloResult(
        mean_price=float(np.mean(prices)),
        median_price=float(np.median(prices)),
        std_price=float(np.std(prices)),
        percentile_5=float(np.percentile(prices, 5)),
        percentile_25=float(np.percentile(prices, 25)),
        percentile_75=float(np.percentile(prices, 75)),
        percentile_95=float(np.percentile(prices, 95)),
        n_samples=n_samples,
        prices=prices,
    )
