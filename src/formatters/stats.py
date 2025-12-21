"""
Stat value formatting by module type.
"""

from src.config.loader import load_attributes

_attributes = load_attributes()

# Attribute IDs that use percentage formatting
PERCENTAGE_ATTRS = {
    _attributes['dda_damage'],  # DDA damage bonus
}

# Attribute IDs that use HP formatting
HP_ATTRS = {
    _attributes['shield_cap'],     # Shield extender HP
    _attributes['armor_hp'],       # Armor plate HP
    _attributes['shield_boost'],   # Shield booster amount
    _attributes['cap_capacity'],   # Cap battery capacity
}

# Attribute IDs that use range formatting (km)
RANGE_ATTRS = {
    _attributes['range'],  # Optimal range
}

# Module types that use multiplier formatting
MULTIPLIER_TYPES = {'dps', 'bcs'}


def format_stat(value: float, module_type: str, attr_id: int = None) -> str:
    """
    Format stat value based on module type and attribute.

    Args:
        value: Stat value
        module_type: Type of module (dps, web, dda, etc.)
        attr_id: Optional attribute ID for more specific formatting

    Returns:
        Formatted stat string
    """
    if attr_id is not None:
        if attr_id in PERCENTAGE_ATTRS:
            return f"{value:.2f}%"
        if attr_id in HP_ATTRS:
            return f"{value:.0f} HP"
        if attr_id in RANGE_ATTRS:
            return f"{value/1000:.1f}km"

    # Module type based formatting
    if module_type in MULTIPLIER_TYPES:
        return f"{value:.3f}x"
    elif module_type == 'dda':
        return f"{value:.2f}%"
    elif module_type == 'shield_extender':
        return f"{value:.0f} HP"
    elif module_type == 'cap_battery':
        return f"{value:.0f} GJ"
    elif module_type == 'armor_plate':
        return f"{value:.0f} HP"
    elif module_type == 'web':
        return f"{abs(value):.1f}%"
    elif module_type == 'afterburner':
        return f"{value:.1f}%"
    elif module_type == 'warp_disruptor':
        return f"{value/1000:.1f}km"
    elif module_type == 'shield_booster':
        return f"{value:.1f} HP/s"
    else:
        return f"{value:.2f}"


def format_stat_range(min_val: float, max_val: float, module_type: str) -> str:
    """
    Format a stat range.

    Args:
        min_val: Minimum value
        max_val: Maximum value
        module_type: Type of module

    Returns:
        Formatted range string (e.g., "[1.12x -> 1.14x]")
    """
    min_str = format_stat(min_val, module_type)
    max_str = format_stat(max_val, module_type)
    return f"[{min_str} -> {max_str}]"
