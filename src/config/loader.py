"""
Configuration loader for EVE Online Mutated Module Analyzer.

Loads and validates YAML configuration files for roll targets, attributes, and constants.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import yaml


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


@dataclass
class MutaplasmidRange:
    """Roll range for a mutaplasmid attribute."""
    attr_id: int
    min_mult: float
    max_mult: float
    high_is_good: bool = True


@dataclass
class RollTarget:
    """Configuration for a roll target with success criteria."""
    key: str
    name: str
    base_type_id: int
    base_name: str
    muta_type_id: int
    muta_name: str
    base_stats: dict = field(default_factory=dict)
    muta_ranges: list = field(default_factory=list)
    primary_desc: str = ""
    secondary_desc: str = ""
    module_type: str = "dps"


# Cached configuration data
_attributes: Optional[dict] = None
_constants: Optional[dict] = None
_targets: Optional[dict] = None


def load_attributes() -> dict[str, int]:
    """
    Load attribute name to ID mappings from config/attributes.yaml.

    Returns:
        Dict mapping attribute names (e.g., 'damage') to EVE dogma IDs (e.g., 64)
    """
    global _attributes
    if _attributes is not None:
        return _attributes

    attrs_path = CONFIG_DIR / "attributes.yaml"
    if not attrs_path.exists():
        raise FileNotFoundError(f"Attributes config not found: {attrs_path}")

    with open(attrs_path, 'r') as f:
        data = yaml.safe_load(f)

    _attributes = data.get('attributes', {})
    return _attributes


def load_constants() -> dict:
    """
    Load constants from config/constants.yaml.

    Returns:
        Dict with API URLs, cache settings, simulation parameters, etc.
    """
    global _constants
    if _constants is not None:
        return _constants

    constants_path = CONFIG_DIR / "constants.yaml"
    if not constants_path.exists():
        raise FileNotFoundError(f"Constants config not found: {constants_path}")

    with open(constants_path, 'r') as f:
        _constants = yaml.safe_load(f)

    return _constants


def _resolve_attr_id(attr_name: str, attributes: dict[str, int]) -> int:
    """Resolve attribute name to ID."""
    if attr_name not in attributes:
        raise ValueError(f"Unknown attribute name: {attr_name}")
    return attributes[attr_name]


def _load_targets_from_yaml(yaml_path: Path, attributes: dict[str, int]) -> list[RollTarget]:
    """Load roll targets from a single YAML file."""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    targets = []
    for item in data.get('targets', []):
        # Convert base_stats keys from names to IDs
        base_stats_raw = item.get('base_stats', {})
        base_stats = {}
        for attr_name, value in base_stats_raw.items():
            attr_id = _resolve_attr_id(attr_name, attributes)
            base_stats[attr_id] = value

        # Convert muta_ranges to MutaplasmidRange objects with resolved IDs
        muta_ranges = []
        for range_data in item.get('muta_ranges', []):
            attr_id = _resolve_attr_id(range_data['attr'], attributes)
            muta_ranges.append(MutaplasmidRange(
                attr_id=attr_id,
                min_mult=range_data['min_mult'],
                max_mult=range_data['max_mult'],
                high_is_good=range_data.get('high_is_good', True),
            ))

        target = RollTarget(
            key=item['key'],
            name=item['name'],
            base_type_id=item['base_type_id'],
            base_name=item['base_name'],
            muta_type_id=item['muta_type_id'],
            muta_name=item['muta_name'],
            base_stats=base_stats,
            muta_ranges=muta_ranges,
            primary_desc=item.get('primary_desc', ''),
            secondary_desc=item.get('secondary_desc', ''),
            module_type=item.get('module_type', 'dps'),
        )
        targets.append(target)

    return targets


def load_all_targets() -> dict[str, RollTarget]:
    """
    Load all roll targets from config/targets/*.yaml files.

    Returns:
        Dict mapping target keys (e.g., 'in_heatsink_unstable') to RollTarget objects
    """
    global _targets
    if _targets is not None:
        return _targets

    attributes = load_attributes()
    targets_dir = CONFIG_DIR / "targets"

    if not targets_dir.exists():
        raise FileNotFoundError(f"Targets directory not found: {targets_dir}")

    _targets = {}
    for yaml_file in targets_dir.glob("*.yaml"):
        try:
            targets = _load_targets_from_yaml(yaml_file, attributes)
            for target in targets:
                if target.key in _targets:
                    raise ValueError(f"Duplicate target key '{target.key}' in {yaml_file}")
                _targets[target.key] = target
        except Exception as e:
            raise ValueError(f"Error loading {yaml_file}: {e}") from e

    return _targets


def get_target(key: str) -> RollTarget:
    """Get a specific roll target by key."""
    targets = load_all_targets()
    if key not in targets:
        available = ', '.join(sorted(targets.keys()))
        raise KeyError(f"Unknown target '{key}'. Available: {available}")
    return targets[key]


def clear_cache():
    """Clear cached configuration data (useful for testing)."""
    global _attributes, _constants, _targets
    _attributes = None
    _constants = None
    _targets = None


# Convenience: export attribute IDs as module-level constants
def get_attr_id(name: str) -> int:
    """Get attribute ID by name."""
    return load_attributes()[name]
