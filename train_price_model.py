#!/usr/bin/env python3
"""
Train XGBoost price prediction models for mutated modules.

CLI wrapper for the src.models training infrastructure.
Trains models that can be used by find_prices.py and roi_calculator.py.

Usage:
    # Train cap battery model with default (90 days)
    python train_price_model.py

    # Train with 6 months of data
    python train_price_model.py --days 180

    # Train a specific module type
    python train_price_model.py --module gyro --days 180

    # List all available module types
    python train_price_model.py --list-types

    # List trained models
    python train_price_model.py --list-models
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.models import (
    MODULE_CONFIGS,
    get_or_train_model,
    list_available_models,
    predict_price,
)


def format_isk(value: float) -> str:
    """Format ISK value with suffix."""
    if value >= 1_000_000_000:
        return f"{value/1_000_000_000:.2f}B"
    elif value >= 1_000_000:
        return f"{value/1_000_000:.2f}M"
    elif value >= 1_000:
        return f"{value/1_000:.2f}K"
    return f"{value:.0f}"


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train XGBoost price prediction models for mutated modules"
    )
    parser.add_argument(
        '--module', '-m',
        type=str,
        default='capbat',
        choices=list(MODULE_CONFIGS.keys()),
        help='Module type to train (default: capbat)'
    )
    parser.add_argument(
        '--days', '-d',
        type=int,
        default=90,
        help='Days of historical data to gather (default: 90)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force retraining even if model exists'
    )
    parser.add_argument(
        '--list-types',
        action='store_true',
        help='List available module types and exit'
    )
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List trained models and exit'
    )
    parser.add_argument(
        '--predict',
        action='store_true',
        help='After training, predict prices for sample cap batteries'
    )

    args = parser.parse_args()

    if args.list_types:
        print("Available module types:")
        for key, config in MODULE_CONFIGS.items():
            print(f"  {key:12} - {config.name}")
        return

    if args.list_models:
        models = list_available_models()
        if not models:
            print("No trained models found.")
            return

        print("Trained models:")
        print(f"{'Type':12} {'Days':>6} {'Age':>8} {'R²':>8} {'Contracts':>10}")
        print("-" * 50)
        for m in sorted(models, key=lambda x: (x['module_type'], x['days'])):
            print(f"{m['module_type']:12} {m['days']:>6}d {m['age_days']:>7.1f}d {m['r2_score']:>7.3f} {m['n_contracts']:>10}")
        return

    # Train model
    print("=" * 80)
    print(f"Training {args.module} model")
    print("=" * 80)

    model_info = get_or_train_model(
        module_type=args.module,
        days=args.days,
        force_retrain=args.force,
        verbose=True
    )

    if model_info is None:
        print("\nTraining failed!")
        return 1

    print(f"\n{'='*80}")
    print("Training Complete")
    print(f"{'='*80}")
    print(f"  Module type: {model_info.module_type}")
    print(f"  Training days: {model_info.days}")
    print(f"  Contracts used: {model_info.n_contracts}")
    print(f"  R² Score: {model_info.r2_score:.3f}")
    print(f"  MAE: {format_isk(model_info.mae)}")

    # Predict for sample cap batteries if requested
    if args.predict and args.module == 'capbat':
        print(f"\n{'='*80}")
        print("Sample Predictions")
        print(f"{'='*80}")

        batteries = [
            {'id': 1, 'cap_bonus': 1698.61, 'cpu': 51.24, 'powergrid': 278.19, 'cap_warfare_resist': -25.2},
            {'id': 2, 'cap_bonus': 1834.56, 'cpu': 42.53, 'powergrid': 375.97, 'cap_warfare_resist': -25.7},
            {'id': 3, 'cap_bonus': 1669.12, 'cpu': 49.17, 'powergrid': 388.5, 'cap_warfare_resist': -29.8},
            {'id': 4, 'cap_bonus': 2125.03, 'cpu': 41.45, 'powergrid': 353.94, 'cap_warfare_resist': -29.9},
            {'id': 5, 'cap_bonus': 1698.06, 'cpu': 51.59, 'powergrid': 276.9, 'cap_warfare_resist': -27.8},
            {'id': 6, 'cap_bonus': 1767.95, 'cpu': 51.33, 'powergrid': 273.44, 'cap_warfare_resist': -31.9},
            {'id': 7, 'cap_bonus': 2050.23, 'cpu': 39.49, 'powergrid': 295.33, 'cap_warfare_resist': -27.7},
            {'id': 8, 'cap_bonus': 2079.17, 'cpu': 43.58, 'powergrid': 381.44, 'cap_warfare_resist': -25.2},
            {'id': 9, 'cap_bonus': 2067.7, 'cpu': 49.08, 'powergrid': 378.85, 'cap_warfare_resist': -25.9},
            {'id': 10, 'cap_bonus': 1841.66, 'cpu': 38.43, 'powergrid': 380.58, 'cap_warfare_resist': -28.9},
            {'id': 11, 'cap_bonus': 2091.73, 'cpu': 48.96, 'powergrid': 368.62, 'cap_warfare_resist': -30.4},
        ]

        print(f"\n{'#':<4} | {'Cap':>8} | {'CPU':>6} | {'PG':>6} | {'Resist':>7} | {'Predicted':>12}")
        print("-" * 65)

        total = 0
        for bat in sorted(batteries, key=lambda x: -x['cap_bonus']):
            price = predict_price(model_info, bat)
            total += price
            print(f"#{bat['id']:<3} | {bat['cap_bonus']:>8.1f} | {bat['cpu']:>6.1f} | {bat['powergrid']:>6.1f} | {bat['cap_warfare_resist']:>6.1f}% | {format_isk(price):>12}")

        print("-" * 65)
        print(f"{'TOTAL':>51} | {format_isk(total):>12}")

    return 0


if __name__ == '__main__':
    exit(main() or 0)
