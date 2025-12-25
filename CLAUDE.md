# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EVE Online Mutated Module Price Finder - a tool to find the lowest contract prices for equivalent mutated (abyssal) modules by comparing dogma attributes and using XGBoost ML models for price prediction.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Find prices for cap batteries (auto-trains model if needed)
python find_prices.py -m capbat --cap 2000 --cpu 45 --pg 320 --resist -28

# Find prices for gyrostabilizers
python find_prices.py -m gyro -d 1.145 -r 12.49 -c 18.25

# Calculate ROI for all roll targets
python roi_calculator.py

# Calculate ROI for specific target
python roi_calculator.py --target rf_capbat_gravid

# Train XGBoost model manually
python train_price_model.py --module capbat --days 180

# List available module types for training
python train_price_model.py --list-types

# List trained models
python train_price_model.py --list-models

# Analyze contract turnover (market demand)
python contract_turnover.py --days 60
```

## Architecture

### Core Scripts
- `find_prices.py` - Price finder with XGBoost ML predictions (uses 180-day models)
- `roi_calculator.py` - ROI calculator for rolling mutated modules (uses 5-day models)
- `train_price_model.py` - CLI for training XGBoost price prediction models
- `contract_turnover.py` - Analyzes contract turnover to identify high-demand modules

### Source Modules (`src/`)
- `src/models/` - XGBoost price prediction
  - `training.py` - Generic training logic for any module type
  - `xgboost_manager.py` - Model lifecycle (load, save, staleness check, auto-retrain)
- `src/analysis/` - Analysis utilities
  - `monte_carlo.py` - Success rate calculation
  - `regression.py` - Fallback regression pricing
  - `risk.py` - ROI and bankroll risk calculations
- `src/config/` - Configuration loading from YAML
- `src/data/` - Data fetching (EVE Ref, Fuzzwork)
- `src/formatters/` - ISK and stat formatting

### Configuration (`config/`)
- `constants.yaml` - API URLs, cache settings, model settings
- `attributes.yaml` - EVE dogma attribute ID mappings
- `targets/*.yaml` - Roll target definitions per module type

### Trained Models (`models/`)
- Models stored as `{module_type}_{days}d.pkl`
- Auto-retrain when older than 10 days
- Example: `capbat_180d.pkl`, `capbat_5d.pkl`

## XGBoost Price Prediction

The system uses XGBoost ML models trained on historical contract data to predict prices based on module stats.

### Supported Module Types
| Key | Module Type | Features |
|-----|-------------|----------|
| `capbat` | Large Cap Battery | cap_bonus, cpu, powergrid, cap_warfare_resist |
| `gyro` | Gyrostabilizer | damage_modifier, rof_multiplier, cpu |
| `heatsink` | Heat Sink | damage_modifier, rof_multiplier, cpu |
| `magstab` | Magnetic Field Stabilizer | damage_modifier, rof_multiplier, cpu |
| `bcs` | Ballistic Control System | damage_modifier, rof_multiplier, cpu |
| `entropic` | Entropic Radiation Sink | damage_modifier, rof_multiplier, cpu |

### Model Settings (config/constants.yaml)
```yaml
models:
  model_dir: "models"           # Where trained models are stored
  max_model_age_days: 10        # Auto-retrain after this many days
  find_prices_training_days: 180  # 6 months for find_prices.py
  roi_calculator_training_days: 5  # 5 days for roi_calculator.py (faster)
```

## Data Sources

- [EVE Ref Public Contract Snapshots](https://data.everef.net/public-contracts/) - Contract and mutated item data
- [Fuzzwork Market API](https://market.fuzzwork.co.uk/) - Jita market prices
- [Fuzzwork SDE](https://www.fuzzwork.co.uk/dump/) - Type names and attributes

## Key EVE Dogma Attribute IDs

- `64` - damageMultiplier (Damage Modifier)
- `204` - speedMultiplier (Rate of Fire)
- `50` - cpu (CPU Usage)
- `30` - power (Powergrid)
- `67` - capacitorCapacity (Cap Bonus)
- `2267` - oaCapacitorNeedMultiplier (Cap Warfare Resist)
