import numpy as np

TRADING_DAYS = 252  # trading days in one year, used to turn daily numbers into yearly ones


def total_return(values):
    """How much the money grew, in % (20.0 = +20%)."""
    return (values[-1] / values[0] - 1) * 100


def max_drawdown(values):
    """The worst fall from a high point to a later low, in % (10.0 = fell 10%)."""
    values = np.asarray(values, dtype=np.float64)
    peaks = np.maximum.accumulate(values)  # highest value seen so far, on each day
    falls = (peaks - values) / peaks
    return falls.max() * 100


def sharpe_ratio(values):
    """Return per unit of risk, per year. Higher = smoother growth. Risk-free rate taken as 0."""
    values = np.asarray(values, dtype=np.float64)
    daily = values[1:] / values[:-1] - 1
    if daily.std() == 0:
        return 0.0
    return daily.mean() / daily.std() * np.sqrt(TRADING_DAYS)


def summary(values):
    """All the scores for one list of daily portfolio values."""
    return {
        "final_value": round(float(values[-1]), 2),
        "total_return_pct": round(float(total_return(values)), 2),
        "max_drawdown_pct": round(float(max_drawdown(values)), 2),
        "sharpe_ratio": round(float(sharpe_ratio(values)), 2),
    }
