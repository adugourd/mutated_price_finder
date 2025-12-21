"""
ISK value formatting utilities.
"""


def format_isk(value: float) -> str:
    """
    Format ISK value with appropriate suffix (K, M, B).

    Args:
        value: ISK amount

    Returns:
        Formatted string (e.g., "1.50B", "250.0M", "15.0K")
    """
    if value < 0:
        return f"-{format_isk(abs(value))}"

    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    elif value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    elif value >= 1_000:
        return f"{value / 1_000:.1f}K"
    else:
        return f"{value:.0f}"


def format_isk_full(value: float) -> str:
    """
    Format ISK value with full precision and commas.

    Args:
        value: ISK amount

    Returns:
        Formatted string with thousand separators (e.g., "1,500,000,000")
    """
    return f"{value:,.0f}"
