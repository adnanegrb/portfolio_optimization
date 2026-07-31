"""End-to-end pipeline: fetch data, optimize the portfolio, run VaR, plot everything.

Run with: python main.py
"""

import numpy as np

from portfolio_optimization.data.market_data import load_portfolio_data
from portfolio_optimization.optimization.markowitz import MarkowitzOptimizer
from portfolio_optimization.risk.engine import RiskEngine
from portfolio_optimization.utils.visualization import (
    plot_efficient_frontier,
    plot_var_distribution,
    plot_var_by_method,
    plot_weight_sensitivity,
)

TICKERS = ["AIR.PA", "BNP.PA", "OR.PA", "SAN.PA", "TTE.PA"]  # Airbus, BNP, L'Oréal, Sanofi, TotalEnergies
START = "2021-01-01"
END = "2026-01-01"


def main():
    print("Fetching CAC 40 price data...")
    data = load_portfolio_data(TICKERS, START, END)

    print("Optimizing portfolio (Markowitz mean-variance)...")
    optimizer = MarkowitzOptimizer(data, risk_free_rate=0.02)
    max_sharpe = optimizer.max_sharpe_portfolio()
    min_vol = optimizer.min_volatility_portfolio()
    frontier = optimizer.efficient_frontier(n_points=50)

    print(f"\nMax Sharpe portfolio")
    print(f"  Return: {max_sharpe.expected_return:.2%}")
    print(f"  Volatility: {max_sharpe.volatility:.2%}")
    print(f"  Sharpe ratio: {max_sharpe.sharpe_ratio:.2f}")
    print(f"  Weights: {max_sharpe.as_dict(data.tickers)}")

    print("\nRunning VaR analysis (historical, parametric, Monte Carlo)...")
    engine = RiskEngine(data, max_sharpe.weights, n_simulations=10_000)
    summary = engine.summary(confidence_levels=(0.95, 0.99))
    print(summary.to_string(index=False))

    print("\nGenerating plots...")
    plot_efficient_frontier(frontier, max_sharpe, min_vol, data.tickers, save_path="frontier.png")

    portfolio_returns = data.returns @ max_sharpe.weights
    var_95 = engine.historical.compute(0.95)
    var_99 = engine.historical.compute(0.99)
    plot_var_distribution(portfolio_returns, var_95, var_99, save_path="var_distribution.png")

    plot_var_by_method(summary, save_path="var_by_method.png")
    plot_weight_sensitivity(data, max_sharpe.weights, data.tickers, save_path="sensitivity.png")

    print("Done. Plots saved to the working directory.")


if __name__ == "__main__":
    main()
