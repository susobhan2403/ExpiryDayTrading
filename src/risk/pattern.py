import pandas as pd


def adjust_for_pattern_conflict(momentum: pd.Series, pattern: pd.Series, base_stop: float) -> pd.Series:
    """Tighten stops when momentum and pattern disagree.

    Parameters
    ----------
    momentum : Series of bool, True if momentum bullish.
    pattern : Series of bool, True if pattern bullish.
    base_stop : float representing base stop distance.
    """
    conflict = momentum.ne(pattern)
    stops = pd.Series(base_stop, index=momentum.index)
    stops[conflict] = base_stop * 0.5
    return stops
