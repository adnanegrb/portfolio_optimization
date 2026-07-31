"""Markowitz mean-variance optimization: efficient frontier and max Sharpe portfolio."""

import numpy as np
from scipy.optimize import minimize

from portfolio_optimization.base import PortfolioData, Portfolio


class MarkowitzOptimizer:
    """Solves for the efficient frontier and the tangency (max Sharpe) portfolio.

    Weights are constrained to sum to 1 with no short selling, which matches
    a realistic long-only equity mandate.
    """

    def __init__(self, data: PortfolioData, risk_free_rate: float = 0.02):
        self.data = data
        self.risk_free_rate = risk_free_rate
        self.n = data.n_assets()

    def _portfolio_stats(self, weights: np.ndarray) -> tuple[float, float]:
        ret = float(weights @ self.data.mean_returns)
        vol = float(np.sqrt(weights @ self.data.cov_matrix @ weights))
        return ret, vol

    def _neg_sharpe(self, weights: np.ndarray) -> float:
        ret, vol = self._portfolio_stats(weights)
        return -(ret - self.risk_free_rate) / vol

    def max_sharpe_portfolio(self) -> Portfolio:
        """Finds the portfolio that maximizes the Sharpe ratio on the frontier."""
        bounds = tuple((0.0, 1.0) for _ in range(self.n))
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        x0 = np.repeat(1.0 / self.n, self.n)

        result = minimize(
            self._neg_sharpe, x0, method="SLSQP",
            bounds=bounds, constraints=constraints,
        )
        weights = result.x
        ret, vol = self._portfolio_stats(weights)
        sharpe = (ret - self.risk_free_rate) / vol
        return Portfolio(weights=weights, expected_return=ret, volatility=vol, sharpe_ratio=sharpe)

    def min_volatility_portfolio(self) -> Portfolio:
        """Finds the global minimum variance portfolio."""
        bounds = tuple((0.0, 1.0) for _ in range(self.n))
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        x0 = np.repeat(1.0 / self.n, self.n)

        result = minimize(
            lambda w: self._portfolio_stats(w)[1], x0, method="SLSQP",
            bounds=bounds, constraints=constraints,
        )
        weights = result.x
        ret, vol = self._portfolio_stats(weights)
        sharpe = (ret - self.risk_free_rate) / vol
        return Portfolio(weights=weights, expected_return=ret, volatility=vol, sharpe_ratio=sharpe)

    def efficient_frontier(self, n_points: int = 50) -> list[Portfolio]:
        """Sweeps target returns and finds the minimum-variance portfolio for each."""
        min_ret = self.data.mean_returns.min()
        max_ret = self.data.mean_returns.max()
        target_returns = np.linspace(min_ret, max_ret, n_points)

        bounds = tuple((0.0, 1.0) for _ in range(self.n))
        frontier = []

        for target in target_returns:
            constraints = (
                {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
                {"type": "eq", "fun": lambda w, t=target: w @ self.data.mean_returns - t},
            )
            x0 = np.repeat(1.0 / self.n, self.n)
            result = minimize(
                lambda w: self._portfolio_stats(w)[1], x0, method="SLSQP",
                bounds=bounds, constraints=constraints,
            )
            if result.success:
                ret, vol = self._portfolio_stats(result.x)
                sharpe = (ret - self.risk_free_rate) / vol
                frontier.append(Portfolio(result.x, ret, vol, sharpe))

        return frontier
