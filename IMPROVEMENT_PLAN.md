# Implementation Plan: Repository Improvements

This document outlines a phased approach to improving the codebase quality, maintainability, and robustness of the EVE Online Mutated Module Analyzer.

---

## Phase 1: Configuration Extraction

**Goal:** Remove hardcoded configuration from Python code and establish a clean data/logic separation.

### 1.1 Create Configuration Directory Structure

```
config/
├── targets/
│   ├── shield_extenders.yaml
│   ├── heat_sinks.yaml
│   ├── mag_stabs.yaml
│   ├── gyrostabilizers.yaml
│   ├── webs.yaml
│   ├── ddas.yaml
│   ├── armor_plates.yaml
│   ├── shield_boosters.yaml
│   ├── afterburners.yaml
│   ├── cap_batteries.yaml
│   ├── warp_disruptors.yaml
│   ├── damage_controls.yaml
│   └── armor_repairers.yaml
├── attributes.yaml        # Dogma attribute ID mappings
├── module_types.yaml      # Module type definitions for find_prices.py
└── constants.yaml         # API URLs, cache durations, simulation parameters
```

### 1.2 Define YAML Schema for Roll Targets

Each target configuration file should follow this structure:

```yaml
# Example: config/targets/heat_sinks.yaml
targets:
  - key: in_heatsink_unstable
    name: "Imperial Navy Heat Sink + Unstable"
    base_type_id: 15810
    base_name: "Imperial Navy Heat Sink"
    muta_type_id: 49729
    muta_name: "Unstable Heat Sink Mutaplasmid"
    module_type: dps
    base_stats:
      damage: 1.12        # attr_id: 64
      rof: 0.89           # attr_id: 204
      cpu: 20             # attr_id: 50
    muta_ranges:
      - attr: damage
        min_mult: 0.98
        max_mult: 1.02
        high_is_good: true
      - attr: rof
        min_mult: 0.975
        max_mult: 1.025
        high_is_good: true
      - attr: cpu
        min_mult: 0.8
        max_mult: 1.5
        high_is_good: false
    primary_desc: "DPS > base"
    secondary_desc: "CPU not in worst 10%"
```

### 1.3 Create Configuration Loader Module

**File:** `src/config/loader.py`

```python
# Responsibilities:
# - Load and merge all YAML files from config/targets/
# - Validate schema structure
# - Convert to RollTarget dataclass instances
# - Provide attribute ID resolution (name -> ID mapping)
```

**Functions to implement:**
- `load_all_targets() -> dict[str, RollTarget]`
- `load_attributes() -> dict[str, int]`
- `load_constants() -> dict`
- `load_module_types() -> dict`

### 1.4 Migration Tasks

| Task | Description | Estimated Effort |
|------|-------------|------------------|
| Extract `ROLL_TARGETS` dict | Move ~1000 lines from `roi_calculator.py` to YAML files | 2 hours |
| Extract `MODULE_TYPES` dict | Move from `find_prices.py` to `config/module_types.yaml` | 30 min |
| Extract attribute IDs | Create `config/attributes.yaml` with all `ATTR_*` constants | 30 min |
| Extract API URLs/constants | Move `FUZZWORK_MARKET_URL`, `EVEREF_CONTRACTS_URL`, etc. | 30 min |
| Implement loader | Write `src/config/loader.py` with validation | 2 hours |
| Update imports | Modify all scripts to use config loader | 1 hour |

### 1.5 Acceptance Criteria

- [ ] Zero hardcoded `RollTarget` definitions in Python files
- [ ] All scripts load configuration via `src/config/loader.py`
- [ ] Adding a new roll target requires only YAML changes (no Python edits)
- [ ] Configuration validation fails fast with clear error messages

---

## Phase 2: Module Refactoring

**Goal:** Break the monolithic `roi_calculator.py` into focused, testable modules.

### 2.1 Proposed Package Structure

```
src/
├── __init__.py
├── config/
│   ├── __init__.py
│   └── loader.py
├── data/
│   ├── __init__.py
│   ├── everef.py          # EVE Ref contract data fetching
│   ├── fuzzwork.py        # Fuzzwork market price fetching
│   └── cache.py           # Unified caching logic
├── analysis/
│   ├── __init__.py
│   ├── monte_carlo.py     # Roll simulation
│   ├── regression.py      # Price regression models
│   ├── risk.py            # Binomial risk analysis
│   └── outliers.py        # IQR filtering
├── models/
│   ├── __init__.py
│   ├── targets.py         # RollTarget, MutaplasmidRange dataclasses
│   └── results.py         # Analysis result dataclasses
├── formatters/
│   ├── __init__.py
│   ├── isk.py             # ISK formatting
│   ├── stats.py           # Stat formatting by module type
│   └── console.py         # Console output formatting
└── cli/
    ├── __init__.py
    ├── roi_calculator.py  # CLI entry point (thin wrapper)
    ├── find_prices.py
    ├── contract_turnover.py
    ├── killmail_demand.py
    └── check_hangar.py
```

### 2.2 Module Responsibilities

#### `src/data/everef.py`
```python
# Current location: roi_calculator.py lines 580-650
# Functions to extract:
def download_contract_archive(cache_dir: Path) -> Path
def extract_csv_from_archive(archive_path: Path, csv_name: str) -> pd.DataFrame
def get_contracts_with_items(archive_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]
```

#### `src/data/fuzzwork.py`
```python
# Current location: roi_calculator.py lines 550-580
# Functions to extract:
def get_jita_prices(type_ids: list[int]) -> dict[int, dict]
def get_sell_price(type_id: int) -> float
```

#### `src/analysis/monte_carlo.py`
```python
# Current location: roi_calculator.py lines 1200-1400
# Functions to extract:
def simulate_rolls(target: RollTarget, n_samples: int = 100_000) -> dict[int, np.ndarray]
def calc_dps_success_rate(target: RollTarget) -> SuccessResult
def calc_web_success_rate(target: RollTarget) -> SuccessResult
# ... other module-specific calculators

# Refactoring opportunity: Create a base calculator class or use a registry pattern
# to reduce duplication across calc_*_success_rate functions
```

#### `src/analysis/regression.py`
```python
# Current location: roi_calculator.py lines 650-800
# Functions to extract:
def fit_constrained_regression(stats: np.ndarray, prices: np.ndarray, 
                                base_stat: float, max_stat: float) -> RegressionResult
def estimate_price_at_stat(regression: RegressionResult, stat: float) -> float
```

#### `src/analysis/risk.py`
```python
# Current location: roi_calculator.py lines 1600-1750
# Functions to extract:
def calculate_bankroll_risk(success_rate: float, roll_cost: float, 
                            expected_price: float, bankroll: float) -> RiskResult
def probability_of_profit(n_rolls: int, success_rate: float, 
                          min_successes: int) -> float
```

### 2.3 Refactoring Strategy

**Step 1: Create package structure** (no logic changes)
- Create all directories and `__init__.py` files
- Ensure imports work

**Step 2: Extract data layer**
- Move caching, EVE Ref, and Fuzzwork functions
- Update imports in `roi_calculator.py`
- Verify functionality unchanged

**Step 3: Extract analysis modules**
- Start with `regression.py` (self-contained)
- Then `outliers.py`
- Then `monte_carlo.py` (largest, depends on targets)
- Finally `risk.py`

**Step 4: Extract formatters**
- Move `format_isk()`, `format_stat()`
- Create console output module

**Step 5: Reduce CLI to thin wrapper**
- `cli/roi_calculator.py` should only:
  - Parse arguments
  - Call analysis functions
  - Format and print output

### 2.4 Deduplication: Success Rate Calculators

Current state: 12+ nearly identical `calc_*_success_rate` functions with minor variations.

**Proposed refactor:**

```python
# src/analysis/monte_carlo.py

class SuccessCalculator:
    """Base class for module-specific success calculations."""
    
    def __init__(self, target: RollTarget):
        self.target = target
        self.rolls = simulate_rolls(target)
    
    def primary_success(self) -> np.ndarray:
        """Override: Return boolean array of primary stat success."""
        raise NotImplementedError
    
    def secondary_ok(self) -> np.ndarray:
        """Override: Return boolean array of secondary stat acceptability."""
        raise NotImplementedError
    
    def calculate(self) -> SuccessResult:
        primary = self.primary_success()
        secondary = self.secondary_ok()
        combined = primary & secondary
        return SuccessResult(
            success_rate=np.mean(combined),
            p_primary=np.mean(primary),
            p_secondary=np.mean(secondary),
        )

class DPSCalculator(SuccessCalculator):
    """DPS modules: damage * rof > base."""
    
    def primary_success(self) -> np.ndarray:
        base_dps = self.target.base_stats[ATTR_DAMAGE] / self.target.base_stats[ATTR_ROF]
        rolled_dps = (self.target.base_stats[ATTR_DAMAGE] * self.rolls[ATTR_DAMAGE]) / \
                     (self.target.base_stats[ATTR_ROF] * self.rolls[ATTR_ROF])
        return rolled_dps > base_dps
    
    def secondary_ok(self) -> np.ndarray:
        base_cpu = self.target.base_stats[ATTR_CPU]
        rolled_cpu = base_cpu * self.rolls[ATTR_CPU]
        threshold = np.percentile(rolled_cpu, 90)
        return rolled_cpu < threshold

# Registry for module type -> calculator class
CALCULATORS = {
    'dps': DPSCalculator,
    'web': WebCalculator,
    'shield_booster': ShieldBoosterCalculator,
    # ...
}

def get_success_rate(target: RollTarget) -> SuccessResult:
    calculator_cls = CALCULATORS.get(target.module_type, DPSCalculator)
    return calculator_cls(target).calculate()
```

### 2.5 Acceptance Criteria

- [ ] `roi_calculator.py` reduced from ~2000 lines to <200 lines
- [ ] Each module in `src/analysis/` is independently importable and testable
- [ ] No circular imports
- [ ] All existing CLI functionality preserved
- [ ] Success rate calculators share common base implementation

---

## Phase 3: Type Hints and Pydantic Models

**Goal:** Add comprehensive type annotations and runtime validation.

### 3.1 Add Type Hints to All Modules

**Priority order:**
1. `src/models/` — Define all dataclasses with types first
2. `src/analysis/` — Core computation modules
3. `src/data/` — Data fetching modules
4. `src/formatters/` — Output formatting
5. `src/cli/` — Entry points

**Example transformations:**

```python
# Before
def format_isk(value):
    if value >= 1_000_000_000:
        return f"{value/1_000_000_000:.2f}B"
    ...

# After
def format_isk(value: float) -> str:
    if value >= 1_000_000_000:
        return f"{value/1_000_000_000:.2f}B"
    ...
```

```python
# Before
def simulate_rolls(target):
    rolls = {}
    for muta_range in target.muta_ranges:
        ...
    return rolls

# After
def simulate_rolls(
    target: RollTarget, 
    n_samples: int = NUM_SAMPLES
) -> dict[int, np.ndarray]:
    rolls: dict[int, np.ndarray] = {}
    for muta_range in target.muta_ranges:
        ...
    return rolls
```

### 3.2 Convert Dataclasses to Pydantic Models

**File:** `src/models/targets.py`

```python
from pydantic import BaseModel, Field, field_validator
from typing import Callable, Optional

class MutaplasmidRange(BaseModel):
    """Roll range for a mutaplasmid attribute."""
    attr_id: int = Field(..., gt=0, description="EVE dogma attribute ID")
    min_mult: float = Field(..., gt=0, lt=2, description="Minimum roll multiplier")
    max_mult: float = Field(..., gt=0, lt=2, description="Maximum roll multiplier")
    high_is_good: bool = Field(default=True, description="True if higher values are better")
    
    @field_validator('max_mult')
    @classmethod
    def max_greater_than_min(cls, v, info):
        if 'min_mult' in info.data and v <= info.data['min_mult']:
            raise ValueError('max_mult must be greater than min_mult')
        return v

class RollTarget(BaseModel):
    """Configuration for a roll target with success criteria."""
    name: str = Field(..., min_length=1)
    base_type_id: int = Field(..., gt=0)
    base_name: str
    muta_type_id: int = Field(..., gt=0)
    muta_name: str
    base_stats: dict[int, float] = Field(default_factory=dict)
    muta_ranges: list[MutaplasmidRange] = Field(default_factory=list)
    module_type: str = Field(default="dps")
    primary_desc: str = Field(default="")
    secondary_desc: str = Field(default="")
    
    @field_validator('module_type')
    @classmethod
    def valid_module_type(cls, v):
        valid_types = {'dps', 'bcs', 'web', 'shield_booster', 'armor_plate', 
                       'cap_battery', 'afterburner', 'warp_disruptor', 'dda', 
                       'shield_extender', 'damage_control', 'armor_repairer'}
        if v not in valid_types:
            raise ValueError(f'module_type must be one of {valid_types}')
        return v

    model_config = {
        "frozen": True  # Immutable after creation
    }
```

**File:** `src/models/results.py`

```python
from pydantic import BaseModel, Field
from typing import Optional

class SuccessResult(BaseModel):
    """Result of Monte Carlo success rate calculation."""
    success_rate: float = Field(..., ge=0, le=1)
    p_primary: float = Field(..., ge=0, le=1)
    p_secondary: float = Field(..., ge=0, le=1)
    base_stat_value: Optional[float] = None

class RegressionResult(BaseModel):
    """Result of constrained linear regression."""
    n_total: int = Field(..., ge=0)
    n_sellable: int = Field(..., ge=0)
    n_outliers: int = Field(..., ge=0)
    n_used: int = Field(..., ge=0)
    expected_price: float = Field(..., ge=0)
    midpoint_stat: float
    slope: float
    anchor_stat: Optional[float] = None
    anchor_price: Optional[float] = None
    base_stat: float
    max_stat: float
    data_min_stat: Optional[float] = None
    data_max_stat: Optional[float] = None
    coverage_pct: float = Field(..., ge=0, le=100)
    method: str

class RiskResult(BaseModel):
    """Result of binomial risk analysis."""
    n_rolls: int = Field(..., ge=0)
    min_successes_needed: int = Field(..., ge=0)
    breakeven_k: float
    prob_profitable: float = Field(..., ge=0, le=1)
    expected_profit: float
    profit_at_5pct: float
    profit_at_50pct: float
    profit_at_95pct: float

class ROIResult(BaseModel):
    """Complete ROI analysis result for a single target."""
    target_key: str
    base_price: float
    muta_price: float
    roll_cost: float
    success_result: SuccessResult
    regression_result: RegressionResult
    risk_result: RiskResult
    expected_value: float
    expected_profit: float
    roi_pct: float
```

### 3.3 Update Configuration Loader for Pydantic

```python
# src/config/loader.py
import yaml
from pathlib import Path
from src.models.targets import RollTarget, MutaplasmidRange

def load_targets_from_yaml(yaml_path: Path) -> list[RollTarget]:
    """Load and validate roll targets from YAML file."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    
    targets = []
    for item in data.get('targets', []):
        # Convert muta_ranges from dict to MutaplasmidRange
        ranges = [MutaplasmidRange(**r) for r in item.pop('muta_ranges', [])]
        target = RollTarget(muta_ranges=ranges, **item)
        targets.append(target)
    
    return targets
```

### 3.4 Add mypy Configuration

**File:** `pyproject.toml` (add section)

```toml
[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_ignores = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
plugins = ["pydantic.mypy"]

[[tool.mypy.overrides]]
module = ["pandas.*", "numpy.*", "scipy.*", "requests.*"]
ignore_missing_imports = true
```

### 3.5 Acceptance Criteria

- [ ] All functions have complete type annotations
- [ ] `mypy --strict` passes with zero errors
- [ ] All configuration validated at load time via Pydantic
- [ ] Invalid YAML configurations produce clear error messages
- [ ] IDE autocomplete works for all custom types

---

## Phase 4: Test Suite

**Goal:** Establish comprehensive test coverage for all analysis logic.

### 4.1 Test Directory Structure

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures
├── unit/
│   ├── __init__.py
│   ├── test_monte_carlo.py
│   ├── test_regression.py
│   ├── test_risk.py
│   ├── test_outliers.py
│   ├── test_formatters.py
│   └── test_config_loader.py
├── integration/
│   ├── __init__.py
│   ├── test_data_pipeline.py
│   └── test_full_analysis.py
└── fixtures/
    ├── sample_contracts.csv
    ├── sample_dynamic_items.csv
    └── sample_targets.yaml
```

### 4.2 Unit Tests

#### `tests/unit/test_monte_carlo.py`

```python
import numpy as np
import pytest
from src.analysis.monte_carlo import simulate_rolls, DPSCalculator
from src.models.targets import RollTarget, MutaplasmidRange

class TestSimulateRolls:
    """Test roll simulation."""
    
    def test_returns_correct_attributes(self, sample_dps_target):
        """Simulated rolls contain all expected attribute IDs."""
        rolls = simulate_rolls(sample_dps_target, n_samples=1000)
        for muta_range in sample_dps_target.muta_ranges:
            assert muta_range.attr_id in rolls
    
    def test_roll_bounds(self, sample_dps_target):
        """All rolls fall within mutaplasmid bounds."""
        rolls = simulate_rolls(sample_dps_target, n_samples=10000)
        for muta_range in sample_dps_target.muta_ranges:
            roll_values = rolls[muta_range.attr_id]
            assert np.all(roll_values >= muta_range.min_mult)
            assert np.all(roll_values <= muta_range.max_mult)
    
    def test_uniform_distribution(self, sample_dps_target):
        """Rolls are approximately uniformly distributed."""
        rolls = simulate_rolls(sample_dps_target, n_samples=100000)
        for muta_range in sample_dps_target.muta_ranges:
            roll_values = rolls[muta_range.attr_id]
            midpoint = (muta_range.min_mult + muta_range.max_mult) / 2
            # Mean should be close to midpoint for uniform dist
            assert abs(np.mean(roll_values) - midpoint) < 0.01
    
    def test_reproducibility_with_seed(self, sample_dps_target):
        """Setting random seed produces reproducible results."""
        np.random.seed(42)
        rolls1 = simulate_rolls(sample_dps_target, n_samples=100)
        np.random.seed(42)
        rolls2 = simulate_rolls(sample_dps_target, n_samples=100)
        for attr_id in rolls1:
            np.testing.assert_array_equal(rolls1[attr_id], rolls2[attr_id])

class TestDPSCalculator:
    """Test DPS module success calculation."""
    
    def test_success_rate_bounds(self, sample_dps_target):
        """Success rate is between 0 and 1."""
        result = DPSCalculator(sample_dps_target).calculate()
        assert 0 <= result.success_rate <= 1
        assert 0 <= result.p_primary <= 1
        assert 0 <= result.p_secondary <= 1
    
    def test_combined_less_than_individual(self, sample_dps_target):
        """Combined success rate <= min of individual rates."""
        result = DPSCalculator(sample_dps_target).calculate()
        assert result.success_rate <= result.p_primary
        assert result.success_rate <= result.p_secondary
    
    def test_convergence(self, sample_dps_target):
        """Success rate converges with more samples."""
        results = []
        for n in [1000, 10000, 100000]:
            # Would need to modify simulate_rolls to accept n_samples
            result = DPSCalculator(sample_dps_target).calculate()
            results.append(result.success_rate)
        # Variance should decrease (values should converge)
        # This is a weak test; ideally run multiple trials
        assert len(set(round(r, 2) for r in results)) <= 2
```

#### `tests/unit/test_regression.py`

```python
import numpy as np
import pytest
from src.analysis.regression import fit_constrained_regression

class TestConstrainedRegression:
    """Test price regression fitting."""
    
    def test_perfect_linear_data(self):
        """Regression exactly fits perfectly linear data."""
        stats = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
        prices = np.array([100, 150, 200, 250, 300])  # y = 100*x
        result = fit_constrained_regression(stats, prices, base_stat=1.0, max_stat=3.0)
        assert abs(result.slope - 100) < 1
    
    def test_anchor_at_minimum(self):
        """Regression anchors at lowest stat point."""
        stats = np.array([1.2, 1.5, 2.0])
        prices = np.array([120, 150, 200])
        result = fit_constrained_regression(stats, prices, base_stat=1.0, max_stat=2.5)
        assert result.anchor_stat == 1.2
        assert result.anchor_price == 120
    
    def test_filters_below_base(self):
        """Items below base stat are excluded."""
        stats = np.array([0.8, 0.9, 1.0, 1.1, 1.2])  # Two below base
        prices = np.array([50, 60, 100, 110, 120])
        result = fit_constrained_regression(stats, prices, base_stat=1.0, max_stat=1.5)
        assert result.n_sellable == 3  # Only 1.0, 1.1, 1.2
    
    def test_coverage_calculation(self):
        """Coverage percentage correctly computed."""
        stats = np.array([1.1, 1.2, 1.3])  # Covers 10-30% of [1.0, 2.0] range
        prices = np.array([110, 120, 130])
        result = fit_constrained_regression(stats, prices, base_stat=1.0, max_stat=2.0)
        assert 15 < result.coverage_pct < 25  # ~20%
    
    def test_outlier_removal(self):
        """IQR outliers are removed."""
        stats = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
        prices = np.array([100, 110, 120, 130, 10000])  # Last is outlier
        result = fit_constrained_regression(stats, prices, base_stat=1.0, max_stat=2.0)
        assert result.n_outliers == 1
    
    def test_insufficient_data_fallback(self):
        """Falls back to mean with < 2 data points."""
        stats = np.array([1.1])
        prices = np.array([100])
        result = fit_constrained_regression(stats, prices, base_stat=1.0, max_stat=2.0)
        assert result.method in ('simple_mean', 'mean_after_outlier_removal')
```

#### `tests/unit/test_risk.py`

```python
import pytest
from src.analysis.risk import calculate_bankroll_risk, probability_of_profit

class TestProbabilityOfProfit:
    """Test binomial probability calculations."""
    
    def test_certain_success(self):
        """100% success rate means 100% profit probability."""
        prob = probability_of_profit(n_rolls=10, success_rate=1.0, min_successes=5)
        assert prob == 1.0
    
    def test_impossible_requirement(self):
        """Needing more successes than rolls means 0% probability."""
        prob = probability_of_profit(n_rolls=5, success_rate=0.5, min_successes=10)
        assert prob == 0.0
    
    def test_fair_coin(self):
        """50% success needing half is ~50% for large n."""
        prob = probability_of_profit(n_rolls=100, success_rate=0.5, min_successes=50)
        assert 0.45 < prob < 0.55

class TestBankrollRisk:
    """Test bankroll risk analysis."""
    
    def test_positive_roi_expected_profit(self):
        """Positive ROI yields positive expected profit."""
        result = calculate_bankroll_risk(
            success_rate=0.5,
            roll_cost=100,
            expected_price=300,  # 50% * 300 = 150 expected value > 100 cost
            bankroll=1000
        )
        assert result.expected_profit > 0
    
    def test_negative_roi_expected_loss(self):
        """Negative ROI yields negative expected profit."""
        result = calculate_bankroll_risk(
            success_rate=0.1,
            roll_cost=100,
            expected_price=200,  # 10% * 200 = 20 expected value < 100 cost
            bankroll=1000
        )
        assert result.expected_profit < 0
    
    def test_percentile_ordering(self):
        """5th percentile <= median <= 95th percentile."""
        result = calculate_bankroll_risk(
            success_rate=0.5,
            roll_cost=100,
            expected_price=250,
            bankroll=10000
        )
        assert result.profit_at_5pct <= result.profit_at_50pct
        assert result.profit_at_50pct <= result.profit_at_95pct
```

#### `tests/unit/test_formatters.py`

```python
import pytest
from src.formatters.isk import format_isk
from src.formatters.stats import format_stat

class TestFormatISK:
    """Test ISK value formatting."""
    
    @pytest.mark.parametrize("value,expected", [
        (0, "0"),
        (999, "999"),
        (1000, "1.0K"),
        (1500, "1.5K"),
        (1_000_000, "1.0M"),
        (1_500_000, "1.5M"),
        (1_000_000_000, "1.00B"),
        (2_500_000_000, "2.50B"),
    ])
    def test_positive_values(self, value, expected):
        assert format_isk(value) == expected
    
    @pytest.mark.parametrize("value,expected", [
        (-1000, "-1.0K"),
        (-1_000_000, "-1.0M"),
        (-1_000_000_000, "-1.00B"),
    ])
    def test_negative_values(self, value, expected):
        assert format_isk(value) == expected

class TestFormatStat:
    """Test stat value formatting by module type."""
    
    def test_dps_format(self):
        assert format_stat(1.234, 'dps') == "1.234x"
    
    def test_dda_format(self):
        assert format_stat(25.5, 'dda') == "25.50%"
    
    def test_shield_extender_format(self):
        assert format_stat(3000, 'shield_extender') == "3000 HP"
```

### 4.3 Integration Tests

#### `tests/integration/test_full_analysis.py`

```python
import pytest
from pathlib import Path
from src.config.loader import load_all_targets
from src.analysis.monte_carlo import get_success_rate
from src.analysis.regression import fit_constrained_regression
from src.analysis.risk import calculate_bankroll_risk

class TestFullAnalysisPipeline:
    """Test complete analysis flow with real config."""
    
    @pytest.fixture
    def all_targets(self):
        return load_all_targets()
    
    def test_all_targets_have_success_rate(self, all_targets):
        """Every configured target produces a valid success rate."""
        for key, target in all_targets.items():
            result = get_success_rate(target)
            assert 0 < result.success_rate < 1, f"Invalid rate for {key}"
    
    def test_all_module_types_supported(self, all_targets):
        """All module types have working calculators."""
        module_types = set(t.module_type for t in all_targets.values())
        for target in all_targets.values():
            # Should not raise
            result = get_success_rate(target)
            assert result is not None
```

### 4.4 Fixtures

**File:** `tests/conftest.py`

```python
import pytest
from src.models.targets import RollTarget, MutaplasmidRange

@pytest.fixture
def sample_dps_target():
    """Sample DPS module target for testing."""
    return RollTarget(
        name="Test Heat Sink",
        base_type_id=15810,
        base_name="Imperial Navy Heat Sink",
        muta_type_id=49729,
        muta_name="Unstable Heat Sink Mutaplasmid",
        base_stats={64: 1.12, 204: 0.89, 50: 20},
        muta_ranges=[
            MutaplasmidRange(attr_id=64, min_mult=0.98, max_mult=1.02, high_is_good=True),
            MutaplasmidRange(attr_id=204, min_mult=0.975, max_mult=1.025, high_is_good=True),
            MutaplasmidRange(attr_id=50, min_mult=0.8, max_mult=1.5, high_is_good=False),
        ],
        module_type="dps",
        primary_desc="DPS > base",
        secondary_desc="CPU not in worst 10%",
    )

@pytest.fixture
def sample_contract_data():
    """Sample contract DataFrame for testing."""
    import pandas as pd
    return pd.DataFrame({
        'contract_id': [1, 2, 3, 4, 5],
        'price': [100_000_000, 150_000_000, 200_000_000, 180_000_000, 5_000_000_000],
        'type': ['item_exchange'] * 5,
    })
```

### 4.5 pytest Configuration

**File:** `pyproject.toml` (add section)

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
addopts = "-v --tb=short"
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
]
```

### 4.6 Acceptance Criteria

- [ ] `pytest` runs successfully with zero failures
- [ ] Code coverage > 80% for `src/analysis/`
- [ ] All edge cases documented and tested
- [ ] CI pipeline runs tests on every push

---

## Phase 5: Error Handling

**Goal:** Add robust error handling for network failures and invalid data.

### 5.1 Custom Exception Hierarchy

**File:** `src/exceptions.py`

```python
class MutatedPriceFinderError(Exception):
    """Base exception for all project errors."""
    pass

class DataFetchError(MutatedPriceFinderError):
    """Failed to fetch data from external API."""
    pass

class EVERefError(DataFetchError):
    """EVE Ref API error."""
    pass

class FuzzworkError(DataFetchError):
    """Fuzzwork API error."""
    pass

class ConfigurationError(MutatedPriceFinderError):
    """Invalid configuration."""
    pass

class InsufficientDataError(MutatedPriceFinderError):
    """Not enough data for analysis."""
    pass

class CacheError(MutatedPriceFinderError):
    """Cache read/write error."""
    pass
```

### 5.2 Retry Logic for API Calls

**File:** `src/data/http.py`

```python
import time
import requests
from typing import Optional
from src.exceptions import DataFetchError

def fetch_with_retry(
    url: str,
    max_retries: int = 3,
    timeout: int = 30,
    backoff_factor: float = 2.0,
    headers: Optional[dict] = None,
) -> requests.Response:
    """Fetch URL with exponential backoff retry."""
    
    last_error = None
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.exceptions.Timeout as e:
            last_error = e
            wait_time = backoff_factor ** attempt
            print(f"Timeout fetching {url}, retrying in {wait_time}s...")
            time.sleep(wait_time)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code in (500, 502, 503, 504):
                last_error = e
                wait_time = backoff_factor ** attempt
                print(f"Server error {e.response.status_code}, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise DataFetchError(f"HTTP {e.response.status_code}: {url}") from e
        except requests.exceptions.RequestException as e:
            raise DataFetchError(f"Network error: {url}") from e
    
    raise DataFetchError(f"Max retries exceeded for {url}") from last_error
```

### 5.3 Graceful Degradation

```python
# src/data/fuzzwork.py

def get_jita_prices(type_ids: list[int]) -> dict[int, Optional[dict]]:
    """
    Fetch Jita prices, returning None for items that fail.
    
    Does not raise on individual item failures; instead returns
    partial results with None for failed lookups.
    """
    results = {}
    for type_id in type_ids:
        try:
            results[type_id] = _fetch_single_price(type_id)
        except DataFetchError as e:
            print(f"Warning: Could not fetch price for {type_id}: {e}")
            results[type_id] = None
    return results
```

### 5.4 User-Friendly Error Messages

```python
# src/cli/roi_calculator.py

def main():
    try:
        # ... existing logic ...
    except ConfigurationError as e:
        print(f"Configuration error: {e}")
        print("Check your YAML files in config/targets/")
        sys.exit(1)
    except EVERefError as e:
        print(f"Could not fetch EVE Ref data: {e}")
        print("EVE Ref may be temporarily unavailable. Try again later.")
        sys.exit(2)
    except FuzzworkError as e:
        print(f"Could not fetch market prices: {e}")
        print("Fuzzwork API may be temporarily unavailable. Try again later.")
        sys.exit(2)
    except InsufficientDataError as e:
        print(f"Not enough data for analysis: {e}")
        print("Try a different item or wait for more contract data.")
        sys.exit(3)
    except KeyboardInterrupt:
        print("\nAborted by user.")
        sys.exit(130)
```

### 5.5 Acceptance Criteria

- [ ] All external API calls wrapped in try/except
- [ ] Network timeouts have configurable retry logic
- [ ] Partial failures don't crash the entire analysis
- [ ] Error messages suggest actionable next steps
- [ ] Exit codes distinguish error types

---

## Phase 6: Async HTTP (Optional Enhancement)

**Goal:** Improve performance when fetching multiple contract snapshots.

### 6.1 When to Apply

This phase is most valuable for:
- `contract_turnover.py` — Fetches 60+ daily snapshots
- `killmail_demand.py` — Fetches many killmail pages
- Batch operations in general

**Not needed for:**
- `roi_calculator.py` — Single archive download
- `find_prices.py` — Single archive download

### 6.2 Implementation with aiohttp

**File:** `src/data/async_http.py`

```python
import asyncio
import aiohttp
from typing import Optional
from pathlib import Path

async def fetch_multiple_urls(
    urls: list[str],
    max_concurrent: int = 5,
    timeout: int = 30,
) -> dict[str, Optional[bytes]]:
    """
    Fetch multiple URLs concurrently.
    
    Returns dict mapping URL -> content (or None on failure).
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    results = {}
    
    async def fetch_one(session: aiohttp.ClientSession, url: str):
        async with semaphore:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                    if resp.status == 200:
                        results[url] = await resp.read()
                    else:
                        print(f"HTTP {resp.status} for {url}")
                        results[url] = None
            except Exception as e:
                print(f"Error fetching {url}: {e}")
                results[url] = None
    
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_one(session, url) for url in urls]
        await asyncio.gather(*tasks)
    
    return results

def fetch_all_sync(urls: list[str], **kwargs) -> dict[str, Optional[bytes]]:
    """Synchronous wrapper for fetch_multiple_urls."""
    return asyncio.run(fetch_multiple_urls(urls, **kwargs))
```

### 6.3 Updated contract_turnover.py

```python
# Before: Sequential downloads (slow)
for date in date_range:
    snapshot = download_snapshot(date)
    snapshots.append(snapshot)

# After: Parallel downloads (fast)
from src.data.async_http import fetch_all_sync

urls = [get_snapshot_url(date) for date in date_range]
results = fetch_all_sync(urls, max_concurrent=5)
```

### 6.4 Dependencies Update

```
# requirements.txt
aiohttp>=3.8.0
```

### 6.5 Acceptance Criteria

- [ ] `contract_turnover.py` completes 60-day analysis 3x faster
- [ ] Rate limiting respected (max 5 concurrent requests)
- [ ] Failures don't block other downloads
- [ ] Sync API still available for simple scripts

---

## Implementation Timeline

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: Config Extraction | 1 week | None |
| Phase 2: Module Refactoring | 2 weeks | Phase 1 |
| Phase 3: Type Hints + Pydantic | 1 week | Phase 1, 2 |
| Phase 4: Test Suite | 1-2 weeks | Phase 1, 2, 3 |
| Phase 5: Error Handling | 3-4 days | Phase 2 |
| Phase 6: Async HTTP | 2-3 days | Phase 5 |

**Total estimated effort:** 6-8 weeks (part-time)

---

## Recommended Order of Implementation

1. **Phase 1** — Establishes clean config foundation
2. **Phase 3** (partial) — Add Pydantic models early to guide refactoring
3. **Phase 2** — Major refactoring, now with validated models
4. **Phase 3** (complete) — Finish type hints across new modules
5. **Phase 4** — Tests lock in correct behaviour
6. **Phase 5** — Error handling for production robustness
7. **Phase 6** — Optional performance enhancement

This order minimizes rework by establishing data structures before refactoring logic, and tests before error handling (so you can test error paths).
