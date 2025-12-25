# EVE Online Mutated Module Analyzer

A statistical analysis toolkit for evaluating mutated (abyssal) module pricing and calculating expected ROI for speculative item modification in EVE Online.

## About This Project

As a computational biologist, I built this project to demonstrate how statistical modeling skills translate across domains. The same techniques I use for analyzing biological data—**Monte Carlo simulation**, **machine learning (XGBoost)**, **outlier detection**, and **risk quantification**—apply directly to economic decision-making in complex markets.

EVE Online's abyssal module system creates a fascinating probability problem: players can "mutate" items with random stat modifications, creating a market where identical-looking items have vastly different values based on their rolled attributes. This project answers the question: *"Given the probability distributions and current market prices, which items offer positive expected value?"*

## Statistical Methods

### Distribution Assumption

The Monte Carlo simulation assumes a **uniform (flat) distribution** for each mutated attribute. This is based on the documented game mechanics from [EVE University](https://wiki.eveuniversity.org/Abyssal_modules):

> *"Each stat that is modified by the mutaplasmid is modified independently, from a flat random distribution across the possible range."*

This independence property is key: each attribute rolls separately, meaning we can model the joint probability of multiple favorable outcomes as the product of individual probabilities.

### XGBoost + Monte Carlo Price Prediction
- Trains gradient-boosted decision trees on historical contract data
- Simulates 100,000 random rolls within observed stat ranges
- Predicts price for each simulated roll using XGBoost
- Returns full price distribution (mean, median, percentiles) for ROI calculation
- Models trained on-demand and cached for reuse

### Model Configuration
- `find_prices.py` uses 180-day models (6 months of data, higher accuracy)
- `roi_calculator.py` uses 5-day models (faster training for batch analysis)
- Models auto-retrain when older than 10 days
- Supported modules: `capbat`, `gyro`, `heatsink`, `magstab`, `bcs`, `entropic`

### Risk Analysis (Binomial Distribution)
- Models portfolio outcomes given finite bankroll
- Calculates probability of profitability across N trials
- Reports percentile outcomes (5th, 50th, 95th) for risk assessment
- Identifies items meeting confidence thresholds (e.g., 95% probability of profit)

### Outlier Detection
- IQR-based filtering removes price manipulation artifacts
- Separates sellable vs. unsellable items based on stat thresholds

## Tools

### ROI Calculator (`roi_calculator.py`)
The core analysis tool. Evaluates expected return on investment for mutating items.

```bash
python roi_calculator.py                    # Analyze all configured items
python roi_calculator.py -t dg_dda_radical  # Specific item analysis
python roi_calculator.py -b 1000000000      # Risk analysis with 1B ISK bankroll
python roi_calculator.py -b 1000000000 -m 5 # Filter to items needing ≤5 sales
```

**Example Output:**
```
Republic Fleet Large Cap Battery + Gravid
----------------------------------------------------------------------
  COSTS:
    Base item:          28.0M  (Republic Fleet Large Cap Battery)
    Mutaplasmid:        55.0M  (Gravid Large Cap Battery Mutaplasmid)
    Roll cost:          83.0M

  XGBOOST MONTE CARLO (100,000 samples):
    Mean price:                   369.8M
    Median price:                 280.3M
    5th percentile:                26.5M  (pessimistic)
    95th percentile:               1.04B  (optimistic)
    ** Using Monte Carlo mean for ROI **

  SUCCESS PROBABILITY (Monte Carlo):
    Sellable:          59.9%  (stat > base + OK secondary)

  ROI ANALYSIS:
    Exp. sale:           369.8M  (mc_xgboost_capbat_5d)
    Exp. value:          221.5M  per roll
    Exp. profit:         138.6M  per roll
    ROI:                 167.0%
```

### Price Finder (`find_prices.py`)
Finds equivalent or better items and recommends pricing using XGBoost ML prediction. Models are trained on-demand when first requested for a module type.

```bash
python find_prices.py -m capbat --cap 2000 --cpu 45 --pg 320 --resist -28  # Cap batteries
python find_prices.py -m magstab -d 1.139 -r 13.01 -c 28.89                # Mag stabs
python find_prices.py -m gyro -d 1.145 -r 12.49 -c 18.25                   # Gyrostabilizers
```

**Example Output:**
```
PRICING RECOMMENDATION (XGBoost ML Model)
============================================================
Loading magstab price model (180 days of training data)...
Using cached magstab model (age: 0.1 days, R²: 0.818)

  Model R² Score: 0.818
  Model trained on: 2935 contracts

  YOUR STATS:
    Damage: 1.139x | ROF: 13.01% | CPU: 28.89 tf
    DPS Multiplier: 1.3093

  --> RECOMMENDED PRICE: 821.38M
```

### Model Trainer (`train_price_model.py`)
Trains XGBoost models for price prediction.

```bash
python train_price_model.py --module capbat --days 180  # Train 6-month model
python train_price_model.py --list-types                 # Show supported modules
python train_price_model.py --list-models                # Show trained models
```

**Supported Modules:** `capbat`, `gyro`, `heatsink`, `magstab`, `bcs`, `entropic`

### Supporting Tools
- `contract_turnover.py` - Market velocity analysis (identifies high-demand modules)
- `killmail_demand.py` - Demand estimation from combat data
- `find_prices_multi.py` - Batch price queries

## Technical Implementation

**Data Pipeline:**
1. Fetches contract snapshots from EVE Ref (30-min cache)
2. Extracts dogma attributes from compressed CSV data
3. Joins with live market prices via Fuzzwork API
4. Applies XGBoost ML models for price prediction (on-demand training)
5. Uses Monte Carlo simulation for ROI expected value calculation

**Model Management:**
- Models stored in `models/` directory as pickle files
- Auto-retrain when models exceed 10 days age
- `find_prices.py` uses 180-day models (higher accuracy)
- `roi_calculator.py` uses 5-day models (faster training for batch analysis)

**Key Dependencies:** `pandas`, `numpy`, `scipy`, `requests`, `xgboost`, `scikit-learn`

## Installation

```bash
pip install -r requirements.txt
python roi_calculator.py
```

## Data Sources

| Source | Data | Update Frequency |
|--------|------|------------------|
| [EVE Ref](https://data.everef.net/public-contracts/) | Contract snapshots with item attributes | 30 minutes |
| [Fuzzwork](https://market.fuzzwork.co.uk/) | Jita market prices | Real-time |
| [EVE ESI](https://esi.evetech.net/) | Base item statistics | Static |
| [EVE University](https://wiki.eveuniversity.org/Abyssal_modules) | Game mechanics documentation | Reference |

## License

MIT
