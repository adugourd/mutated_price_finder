# EVE Online Mutated Module Tools

A collection of tools for analyzing mutated (abyssal) module prices and calculating ROI for rolling mutaplasmids in EVE Online.

## Installation

```bash
pip install -r requirements.txt
```

Requirements: `requests`, `pandas`, `numpy`, `scipy`

## Tools

### 1. Price Finder (`find_prices.py`)

Find the lowest contract prices for mutated modules with equivalent or better stats than yours.

```bash
# Run with default values (Republic Fleet Gyro stats)
python find_prices.py

# Run with custom stats
python find_prices.py -d 1.145 -r 12.49 -c 18.25

# Filter for Republic Fleet Gyro mutations only
python find_prices.py -rf

# Show both worse and better items for comparison
python find_prices.py --show-better
```

### 2. ROI Calculator (`roi_calculator.py`)

Calculate expected ROI for rolling mutated modules based on Monte Carlo simulation and contract price analysis.

```bash
# Analyze all configured items
python roi_calculator.py

# Analyze a specific item
python roi_calculator.py -t dg_dda_radical

# Risk analysis with custom bankroll (default: 1B)
python roi_calculator.py -b 500000000

# Custom confidence threshold (default: 95%)
python roi_calculator.py -b 1000000000 -c 90

# Limit max sales needed to profit (default: 5)
python roi_calculator.py -m 10
```

#### Available targets:
- `dg_dda` - Dread Guristas DDA + Radical
- `in_heatsink` - Imperial Navy Heat Sink + Unstable
- `fn_web` - Fed Navy Stasis Webifier + Unstable
- `rf_gyro` - Republic Fleet Gyrostabilizer + Unstable
- `dom_gyro` - Domination Gyrostabilizer + Unstable
- And more...

#### ROI Methodology

The ROI calculator uses a stat-based pricing methodology:

1. **Success Rate (Monte Carlo)**
   - Simulates 100,000 rolls using mutaplasmid attribute ranges
   - A roll is "sellable" if: `rolled_stat > base_stat` AND `secondary_stat` is not catastrophic
   - Example: For DDA with 0.8-1.2 roll range, ~50% of rolls improve damage

2. **Expected Sale Price (Regression-based)**
   - Fetches ALL contracts with the target mutaplasmid (any source item)
   - Filters to sellable items: `stat > base_stat`
   - Removes price outliers using IQR method
   - Fits linear regression constrained through `(worst_sellable_stat, worst_sellable_price)`
   - Expected price = fitted value at midpoint of `[base_stat, max_stat]` range

3. **Data Coverage Confidence**
   - Measures how much of the stat range `[base_stat, max_stat]` is covered by contract data
   - Coverage = `(max_data_stat - min_data_stat) / (max_stat - base_stat) × 100%`
   - **< 50% coverage**: Flagged as "LOW CONFIDENCE" - regression may be unreliable
   - **≥ 50% coverage**: Sufficient data for reliable price estimation

4. **ROI Formula**
   ```
   roll_cost = base_item_price + mutaplasmid_price
   expected_value = success_rate × expected_sale_price
   expected_profit = expected_value - roll_cost
   ROI = expected_profit / roll_cost × 100%
   ```

   This correctly accounts for total loss on bad rolls:
   - Bad roll (55%): lose entire roll_cost (both mutaplasmid AND base item)
   - Good roll (45%): sell for expected_sale_price

5. **Bankroll Risk Analysis**
   - Given a bankroll B, calculates how many rolls you can afford: `N = floor(B / roll_cost)`
   - Uses binomial distribution to model outcomes after N rolls
   - Calculates probability of being profitable: `P(K >= min_k)` where K is number of successes
   - Minimum successes needed: `min_k = ceil(N × roll_cost / sale_price)`
   - Reports profit percentiles (5th, 50th, 95th) to show risk range
   - Identifies "safe" items with >= 95% probability of profit
   - **Max-sells filter** (`-m`): Excludes items requiring too many successful sales (default: 5)

   **Why this matters**: High-ROI items like DG DDA + Radical (117% ROI) require ~440M per roll.
   With a 1B bankroll, you can only do 2 rolls. Even with 45% success rate, there's a 30% chance
   of losing both rolls.

   Meanwhile, Heat Sink II + Decayed has 985% ROI but needs 23 sales to profit - too time-consuming!
   The max-sells filter finds the sweet spot: items with high confidence AND practical sales volume.

#### Example Output

```
Dread Guristas DDA + Radical
----------------------------------------------------------------------
  COSTS:
    Base item:          88.1M  (Dread Guristas Drone Damage Amplifier)
    Mutaplasmid:       352.4M  (Radical Drone Damage Amplifier Mutaplasmid)
    Roll cost:         440.5M

  CONTRACT ANALYSIS (94 contracts, stat-based):
    Stat range:    [23.8% -> 28.6%]
    Midpoint:      26.18%
    Below base (unsellable):     8 (discarded)
    Price outliers (IQR):        1 (discarded)
    Used for regression:        85
    Anchor point:  (24.03%, 980.0M)
    Slope:         532.8M/% stat
    Data coverage: [24.03% -> 28.56%] = 95%
    Exp. price at midpoint:     2.13B

  SUCCESS PROBABILITY (Monte Carlo):
    Sellable:    45.0%  (stat > base + OK secondary)

  ROI ANALYSIS:
    Exp. value:    957.5M  per roll
    Exp. profit:   516.9M  per roll
    ROI:           117.3%

  VERDICT: EXCELLENT
```

#### Risk Analysis Example (1B bankroll, max 5 sales)

```
======================================================================
RISK ANALYSIS - 1.00B BANKROLL
======================================================================

Items with >= 95% probability of profit AND <= 5 sales to profit:
(Practical constraint: you don't want to roll/sell hundreds of items)

Rank P(Profit)  Rolls  Need    E[Profit]       5th%     Median  ROI/Roll  Item
---------------------------------------------------------------------------------------------------------
1       100.0%     40     5        3.30B      2.11B      3.30B    337.4%  Entropic Sink II + Unstable
2        97.2%      6     1        2.78B     439.8M      3.22B    292.8%  Imperial Navy Heat Sink + Unstable
3        99.4%     11     3        2.00B     835.4M      2.19B    205.7%  Republic Fleet Cap Battery + Gravid
4       100.0%     20     4        1.98B     999.6M      1.98B    204.6%  Imperial Navy 1600mm Plate + Gravid
5        98.6%     11     2        1.82B     138.8M      1.85B    182.0%  Fed Navy Mag Stab + Unstable

  RECOMMENDED: Entropic Sink II + Unstable
  With 1.00B you can do 40 rolls
  Need 5/40 successes to profit (breakeven: 4.11)
  Probability of profit: 100.0%
  Expected profit: 3.30B
  Worst case (5th pct): 2.11B
  Median outcome: 3.30B

----------------------------------------------------------------------
HIGH CONFIDENCE BUT TOO MANY SELLS (need > 5 sales):
----------------------------------------------------------------------
P(Profit)  Rolls  Need  ROI/Roll  Item
   100.0%    416    23    985.3%  Heat Sink II + Decayed
   100.0%     75     8    433.9%  Ballistic Control System II + Gravid

  These are statistically safe but require too many rolls/sales.
  Use -m to increase max-sells if you have time for high-volume trading.

----------------------------------------------------------------------
HIGH ROI BUT RISKY (< 95% confidence with 1.00B):
----------------------------------------------------------------------
P(Profit)  Rolls  Need  ROI/Roll  Profit/Roll  Item
    70.0%      2     1    120.2%       525.1M  Dread Guristas DDA + Radical

  These items have great ROI but you can't do enough rolls with 1.00B
  to reliably overcome RNG variance. Consider a larger bankroll.
```

### 3. Contract Turnover (`contract_turnover.py`)

Analyze contract turnover rates for mutated items.

### 4. Killmail Demand (`killmail_demand.py`)

Analyze demand for mutated modules based on killmail data.

### 5. Multi-Item Price Finder (`find_prices_multi.py`)

Find prices for multiple items at once.

## Data Sources

- **Contract Data**: [EVE Ref Public Contract Snapshots](https://data.everef.net/public-contracts/) - Updated every 30 minutes, cached locally
- **Market Prices**: [Fuzzwork Market API](https://market.fuzzwork.co.uk/) - Jita sell orders
- **Item Stats**: [EVE ESI API](https://esi.evetech.net/) - Base item dogma attributes

## Key EVE Dogma Attribute IDs

| ID | Attribute | Description |
|----|-----------|-------------|
| 64 | damageMultiplier | Damage Modifier (turrets) |
| 204 | speedMultiplier | Rate of Fire |
| 50 | cpu | CPU Usage |
| 1255 | droneDamageBonus | Drone Damage Amplifier bonus |
| 68 | shieldBonus | Shield Booster amount |
| 73 | duration | Cycle time |
| 6 | capacitorNeed | Capacitor usage |

## License

MIT
