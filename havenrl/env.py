import gymnasium as gym
import numpy as np
from gymnasium import spaces

from havenrl.data import TICKERS

WINDOW = 10  # the agent looks at the last 10 days of moves
START_CASH = 100_000.0
COST_RATE = 0.001  # 0.1% fee on every rupee bought or sold

# Each basket is a target split of money, in TICKERS order: [Gold, Liquid, Bond].
BASKETS = np.array([
    [0.6, 0.2, 0.2],  # 0: mostly Gold
    [0.2, 0.2, 0.6],  # 1: mostly Bond
    [0.2, 0.6, 0.2],  # 2: mostly Liquid
])


class HavenEnv(gym.Env):
    """The game board: each step is one trading day where the agent picks a basket."""

    def __init__(self, prices, scaled_returns):
        if list(prices.columns) != TICKERS or not prices.index.equals(scaled_returns.index):
            raise ValueError("prices and scaled_returns must share the same dates and TICKERS columns.")
        self.prices = prices.to_numpy(dtype=np.float64)  # real ₹ prices, used to trade
        self.returns = scaled_returns.to_numpy(dtype=np.float32)  # scaled moves, used to look
        self.dates = prices.index

        n_assets = len(TICKERS)
        self.action_space = spaces.Discrete(len(BASKETS))
        # Observation = last WINDOW days of moves for each asset + weights of [cash, Gold, Liquid, Bond].
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(WINDOW * n_assets + n_assets + 1,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.day = WINDOW - 1  # first day that has a full 10-day window behind it
        self.cash = START_CASH
        self.units = np.zeros(len(TICKERS))  # how many units of each ETF we own
        return self._observation(), self._info()

    def step(self, action):
        today = self.prices[self.day]
        value_before = self._value(today)

        # Rebalance to the chosen basket at today's prices, paying a fee on what is traded.
        traded = np.abs(BASKETS[action] * value_before - self.units * today).sum()
        cost = traded * COST_RATE
        self.units = BASKETS[action] * (value_before - cost) / today
        self.cash = 0.0

        # Move to the next day and see how the portfolio did.
        self.day += 1
        value_after = self._value(self.prices[self.day])
        reward = (value_after / value_before - 1) * 100  # daily % change, after fees

        terminated = self.day == len(self.prices) - 1  # no more days of data
        info = self._info()
        info["cost"] = cost
        return self._observation(), reward, terminated, False, info

    def _value(self, prices):
        return self.cash + (self.units * prices).sum()

    def _weights(self):
        today = self.prices[self.day]
        money = np.concatenate([[self.cash], self.units * today])
        return money / money.sum()

    def _observation(self):
        window = self.returns[self.day - WINDOW + 1 : self.day + 1].flatten()
        return np.concatenate([window, self._weights()]).astype(np.float32)

    def _info(self):
        return {
            "date": self.dates[self.day],
            "value": self._value(self.prices[self.day]),
            "weights": self._weights(),
        }
