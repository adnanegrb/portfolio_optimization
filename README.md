# Portfolio Optimization and VaR Risk Engine

A Markowitz mean-variance optimizer paired with a three-method Value at Risk engine, applied to a five-stock CAC 40 portfolio (Airbus, BNP Paribas, L'Oréal, Sanofi, TotalEnergies). Built to keep the optimizer and the risk engine cleanly separated, so either one can be swapped out or reused on its own.

## What's inside

**Portfolio optimization**
- Markowitz mean-variance optimizer (max Sharpe and min volatility portfolios)
- Efficient frontier construction via constrained quadratic programming
- Long-only, fully-invested constraints (no short selling)

**Risk engine**
- Historical simulation VaR (empirical, no distributional assumption)
- Parametric VaR (variance-covariance method, closed-form under normality)
- Monte Carlo VaR (10,000 correlated simulations via Cholesky decomposition)
- Expected Shortfall (CVaR) alongside every VaR estimate

**Data pipeline**
- Automated price fetching via yfinance
- Daily returns computation and annualization

**Visualizations**
- Efficient frontier with the tangency and minimum-variance portfolios marked
- Portfolio return distribution with VaR thresholds overlaid
- VaR comparison across the three methods and confidence levels
- Sensitivity of portfolio volatility to individual weight shocks

## Quick start

```
pip install -r requirements.txt
python main.py
```

## Optimizing a portfolio

```python
from portfolio_optimization.data.market_data import load_portfolio_data
from portfolio_optimization.optimization.markowitz import MarkowitzOptimizer

tickers = ["AIR.PA", "BNP.PA", "OR.PA", "SAN.PA", "TTE.PA"]
data = load_portfolio_data(tickers, start="2021-01-01", end="2026-01-01")

optimizer = MarkowitzOptimizer(data, risk_free_rate=0.02)
max_sharpe = optimizer.max_sharpe_portfolio()

print(max_sharpe.expected_return, max_sharpe.volatility, max_sharpe.sharpe_ratio)
print(max_sharpe.as_dict(data.tickers))
```

## Running VaR analysis

```python
from portfolio_optimization.risk.engine import RiskEngine

engine = RiskEngine(data, max_sharpe.weights, n_simulations=10_000)
summary = engine.summary(confidence_levels=(0.95, 0.99))
print(summary)
```

Each method also exposes `.expected_shortfall(confidence)` for the average loss beyond the VaR threshold.

## Running tests

```
pytest tests/ -v
```

Tests run on synthetic correlated returns rather than live data, so they stay fast and deterministic. They check that optimized weights sum to one, that the minimum-variance portfolio actually has lower volatility than an equal-weight benchmark, that the efficient frontier is monotonic in return, and that the three VaR methods stay within a reasonable range of each other.

## Project structure

```
portfolio_optimization/
├── base.py                      PortfolioData and Portfolio dataclasses
├── data/
│   └── market_data.py           Price fetching and returns computation
├── optimization/
│   └── markowitz.py             Mean-variance optimizer, efficient frontier
├── risk/
│   ├── historical.py            Empirical VaR
│   ├── parametric.py            Variance-covariance VaR
│   ├── monte_carlo.py           Simulated VaR (Cholesky decomposition)
│   └── engine.py                Combines all three into one summary
└── utils/
    └── visualization.py         Frontier, VaR distribution, sensitivity plots
```

## Math

**Portfolio return and volatility**, given weights $w$, expected returns $\mu$, and covariance matrix $\Sigma$:

$$R_p = w^\top \mu, \qquad \sigma_p = \sqrt{w^\top \Sigma w}$$

**Max Sharpe optimization** solves, subject to $\sum w_i = 1$ and $w_i \geq 0$:

$$\max_w \frac{R_p - r_f}{\sigma_p}$$

**Parametric VaR** assumes portfolio returns are normal, so the loss at confidence level $c$ is:

$$\text{VaR}_c = -(\mu_p + z_{1-c} \, \sigma_p), \qquad z_{1-c} = \Phi^{-1}(1-c)$$

**Monte Carlo VaR** draws correlated asset returns using the Cholesky factor $L$ of $\Sigma$, so that $LL^\top = \Sigma$:

$$r_{\text{sim}} = \mu + Lz, \qquad z \sim \mathcal{N}(0, I)$$

and then takes the empirical percentile of the resulting simulated portfolio returns, exactly as the historical method does on real data.

## License

MIT
