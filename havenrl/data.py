import pandas as pd
import yfinance as yf

ASSETS = {
    "GOLDBEES.NS": "Gold ETF",
    "LIQUIDBEES.NS": "Liquid ETF",
    "EBBETF0430.NS": "Bharat Bond ETF",
}
TICKERS = list(ASSETS)

# Bharat Bond ETF only starts trading on 2019-12-30, so training honestly begins in 2020.
TRAIN_START = "2020-01-01"
TRAIN_END = "2021-12-31"
# 2022 is the "practice exam": never trained on, only used to pick the best saved brain.
VAL_START = "2022-01-01"
VAL_END = "2022-12-31"
TEST_START = "2023-01-01"
TEST_END = "2024-12-31"


def fetch_prices(start=TRAIN_START, end=TEST_END):
    """Download daily close prices for all assets, one column per ticker."""
    # yfinance's `end` is exclusive, so add a day to include TEST_END itself.
    end_inclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(TICKERS, start=start, end=end_inclusive, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError("No data returned from yfinance.")

    # raw has two header rows (Price, Ticker). Taking "Close" leaves one header row:
    # the ticker names. This is the fix for the old "Ticker" dates bug.
    prices = raw["Close"][TICKERS]
    prices = prices.dropna()  # keep only days where all assets have a price
    prices.index.name = "Date"
    prices.columns.name = None
    return prices


def daily_returns(prices):
    """Daily % change of each asset (0.01 = +1%). The first day has no previous day, so it is 0."""
    return prices.pct_change().fillna(0.0)


def split_periods(df):
    """Split any date-indexed table (prices or returns) into the train, validation and test periods."""
    train = df.loc[TRAIN_START:TRAIN_END]
    val = df.loc[VAL_START:VAL_END]
    test = df.loc[TEST_START:TEST_END]
    return train, val, test


def fit_scaler(train_returns):
    """Learn each asset's usual daily move (std) from TRAIN data only, so no future info leaks in."""
    return train_returns.std()


def scale_returns(returns, usual_move):
    """Divide by the usual move so a normal day is about 1 for every asset (Gold, Liquid, Bond)."""
    return returns / usual_move


def prepare_data():
    """Everything train.py and backtest.py need: {"train"/"val"/"test": (real prices, scaled returns)}."""
    prices = fetch_prices()
    returns = daily_returns(prices)  # computed on ALL prices before splitting, so a period's day 1 is not 0
    train_returns, _, _ = split_periods(returns)
    scaled = scale_returns(returns, fit_scaler(train_returns))

    periods = ("train", "val", "test")
    return dict(zip(periods, zip(split_periods(prices), split_periods(scaled))))
