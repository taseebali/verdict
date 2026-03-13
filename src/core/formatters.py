"""Value formatting utilities for human-readable display."""

from typing import Dict, Callable


# Mapping of feature name patterns to formatting functions
# Order matters: check longer/more specific patterns first
FORMAT_MAP: Dict[str, Callable[[float], str]] = {
    'monthlycharge': lambda v: f"${v:.2f}",
    'usage_hours': lambda v: f"{v:.1f} hrs",
    'tenure': lambda v: f"{int(v)} months",
    'monthly': lambda v: f"${v:.2f}",
    'charge': lambda v: f"${v:.2f}",
    'price': lambda v: f"${v:.2f}",
    'payment': lambda v: f"${v:.2f}",
    'income': lambda v: f"${v:.2f}",
    'cost': lambda v: f"${v:.2f}",
    'month': lambda v: f"{int(v)} months",
    'usage': lambda v: f"{v:.1f} hrs",
    'hours': lambda v: f"{v:.1f} hrs",
    'rate': lambda v: f"{v:.1f}%",
    'percent': lambda v: f"{v:.1f}%",
    'ratio': lambda v: f"{v:.1f}%",
    '%': lambda v: f"{v:.1f}%",
    'days': lambda v: f"{int(v)} days",
    'age': lambda v: f"{int(v)} years",
}


def format_value(feature_name: str, value: float) -> str:
    """Convert model values to human-readable format.
    
    Uses a mapping of feature name patterns to format appropriate values.
    
    Args:
        feature_name: Name of the feature to format
        value: Numeric value to format
        
    Returns:
        Formatted string representation of the value
        
    Examples:
        >>> format_value('monthlyCharges', 89.50)
        '$89.50'
        >>> format_value('churnRate', 12.5)
        '12.5%'
        >>> format_value('tenure', 24)
        '24 months'
        >>> format_value('age', 35)
        '35 years'
    """
    feature_lower = feature_name.lower()
    
    # Check against format map
    for key, formatter in FORMAT_MAP.items():
        if key in feature_lower:
            return formatter(value)
    
    # Default formatting
    return f"{value:.2f}"
