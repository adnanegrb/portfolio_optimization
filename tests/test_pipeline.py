"""Sanity checks for the optimizer and risk engine, using synthetic returns.

Synthetic data is used here only because this sandbox has no network access
to Yahoo Finance. The real pipeline (main.py) pulls actual CAC 40 prices.
"""

import numpy as np
import pandas as pd

from portfolio_optimization.base import PortfolioData
from portfolio_optimization.optimization.markowitz import MarkowitzOptimizer
from portfolio_optimization.risk.engine import RiskEngine


def make_synthetic_data(n_assets=5, n_days=1000, seed=7) -> PortfolioData:
    rng = np.random.default_rng(seed)
    tickers = ["AIR.PA", "BNP.PA", "OR.PA", "SAN.PA", "TTE.PA"]

    true_vols = np.array([0.28, 0.30, 0.22, 0.20, 0.26]) / np.sqrt(252)
    true_mean = np.array([0.12, 0.10, 0.09, 0.08, 0.11]) / 252

    corr = np.array([
        [1.00, 0.35, 0.20, 0.15, 0.25],
        [0.35, 1.00, 0.18, 0.12, 0.30],
        [0.20, 0.18, 1.00, 0.22, 0.15],
        [0.15, 0.12, 0.22, 1.00, 0.10],
        [0.25, 0.30, 0.15, 0.10, 1.00],
    ])
    cov = np.outer(true_vols, true_vols) * corr
    chol = np.linalg.cholesky(cov)

    z = rng.standard_normal((n_days, n_assets))
    returns = true_mean + z @ chol.T
    df = pd.DataFrame(returns, columns=tickers)
    return PortfolioData(returns=df)


def test_max_sharpe_weights_sum_to_one():
    data = make_synthetic_data()
    optimizer = MarkowitzOptimizer(data)
    portfolio = optimizer.max_sharpe_portfolio()
    assert abs(portfolio.weights.sum() - 1.0) < 1e-6
    assert (portfolio.weights >= -1e-6).all()


def test_min_vol_has_lower_vol_than_equal_weight():
    data = make_synthetic_data()
    optimizer = MarkowitzOptimizer(data)
    min_vol = optimizer.min_volatility_portfolio()

    equal_weights = np.repeat(1 / data.n_assets(), data.n_assets())
    equal_vol = np.sqrt(equal_weights @ data.cov_matrix @ equal_weights)

    assert min_vol.volatility <= equal_vol


def test_efficient_frontier_is_increasing_in_return():
    data = make_synthetic_data()
    optimizer = MarkowitzOptimizer(data)
    frontier = optimizer.efficient_frontier(n_points=20)
    returns = [p.expected_return for p in frontier]
    assert returns == sorted(returns)


def test_var_increases_with_confidence():
    data = make_synthetic_data()
    weights = np.repeat(1 / data.n_assets(), data.n_assets())
    engine = RiskEngine(data, weights, n_simulations=5000)

    var_95 = engine.historical.compute(0.95)
    var_99 = engine.historical.compute(0.99)
    assert var_99 >= var_95


def test_expected_shortfall_exceeds_var():
    data = make_synthetic_data()
    weights = np.repeat(1 / data.n_assets(), data.n_assets())
    engine = RiskEngine(data, weights, n_simulations=5000)

    var = engine.monte_carlo.compute(0.95)
    es = engine.monte_carlo.expected_shortfall(0.95)
    assert es >= var


def test_three_methods_roughly_agree():
    """The three VaR methods should not diverge wildly on well-behaved synthetic data."""
    data = make_synthetic_data()
    weights = np.repeat(1 / data.n_assets(), data.n_assets())
    engine = RiskEngine(data, weights, n_simulations=10_000)

    summary = engine.summary(confidence_levels=(0.95,))
    values = summary["VaR"].values
    assert values.max() / values.min() < 1.5
