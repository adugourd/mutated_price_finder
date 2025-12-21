# EVE Online Mutated Module Analyzer

A statistical analysis toolkit for evaluating mutated (abyssal) module pricing and calculating expected ROI for speculative item modification in EVE Online.

## About This Project

As a computational biologist, I built this project to demonstrate how statistical modeling skills translate across domains. The same techniques I use for analyzing biological data—**Monte Carlo simulation**, **regression modeling**, **outlier detection**, and **risk quantification**—apply directly to economic decision-making in complex markets.

EVE Online's abyssal module system creates a fascinating probability problem: players can "mutate" items with random stat modifications, creating a market where identical-looking items have vastly different values based on their rolled attributes. This project answers the question: *"Given the probability distributions and current market prices, which items offer positive expected value?"*

## Statistical Methods

### Monte Carlo Simulation
- Simulates 100,000 mutation rolls per item configuration
- Models multi-attribute joint probability (damage, rate of fire, CPU cost)
- Calculates success rates accounting for correlated outcomes

### Regression-Based Price Estimation
- Fits constrained linear models through market data
- Anchors regression at theoretical minimum sellable value
- Quantifies model confidence via data coverage metrics

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
Dread Guristas DDA + Radical
----------------------------------------------------------------------
  COSTS:
    Base item:          88.1M  (Dread Guristas Drone Damage Amplifier)
    Mutaplasmid:       352.4M  (Radical Drone Damage Amplifier Mutaplasmid)
    Roll cost:         440.5M

  CONTRACT ANALYSIS (94 contracts, stat-based):
    Stat range:    [23.8% -> 28.6%]
    Data coverage: 95%
    Exp. price:    2.13B

  SUCCESS PROBABILITY (Monte Carlo, n=100,000):
    Sellable:    45.0%

  ROI ANALYSIS:
    Expected value:   957.5M per roll
    Expected profit:  516.9M per roll
    ROI:              117.3%
```

### Price Finder (`find_prices.py`)
Finds equivalent or better items at lower prices.

```bash
python find_prices.py -d 1.145 -r 12.49 -c 18.25  # Custom stats
python find_prices.py -rf                          # Filter by source type
python find_prices.py --show-better                # Include superior items
```

### Supporting Tools
- `contract_turnover.py` - Market velocity analysis
- `killmail_demand.py` - Demand estimation from combat data
- `find_prices_multi.py` - Batch price queries

## Technical Implementation

**Data Pipeline:**
1. Fetches contract snapshots from EVE Ref (30-min cache)
2. Extracts dogma attributes from compressed CSV data
3. Joins with live market prices via Fuzzwork API
4. Applies statistical models for valuation

**Key Dependencies:** `pandas`, `numpy`, `scipy`, `requests`

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

## License

MIT
