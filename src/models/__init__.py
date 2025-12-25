"""
XGBoost price prediction models for mutated modules.
"""

from src.models.training import (
    MODULE_CONFIGS,
    ModuleTypeConfig,
    TrainingResult,
    train_model,
    predict_price,
)

from src.models.xgboost_manager import (
    ModelInfo,
    MonteCarloResult,
    get_or_train_model,
    load_model,
    save_model,
    model_exists,
    get_model_age,
    list_available_models,
    delete_model,
    predict_price as predict_with_model,
    simulate_prices_monte_carlo,
)

__all__ = [
    # Training
    'MODULE_CONFIGS',
    'ModuleTypeConfig',
    'TrainingResult',
    'train_model',
    'predict_price',
    # Model management
    'ModelInfo',
    'MonteCarloResult',
    'get_or_train_model',
    'load_model',
    'save_model',
    'model_exists',
    'get_model_age',
    'list_available_models',
    'delete_model',
    'predict_with_model',
    'simulate_prices_monte_carlo',
]
