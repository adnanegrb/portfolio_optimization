# Portfolio Optimization and VaR Risk Engine

![Language](https://img.shields.io/badge/Language-Python-blue) ![Topic](https://img.shields.io/badge/Topic-Portfolio%20Optimization-purple) ![Domain](https://img.shields.io/badge/Domain-Quantitative%20Finance-darkblue) ![Methods](https://img.shields.io/badge/Methods-Markowitz%20%7C%20Historical%20VaR%20%7C%20Parametric%20VaR%20%7C%20Monte%20Carlo%20VaR-orange)

A Markowitz mean-variance optimizer paired with a three-method VaR engine, applied to five CAC 40 stocks (Airbus, BNP Paribas, L'Oréal, Sanofi, TotalEnergies). I kept the optimizer and the risk engine as two separate pieces on purpose, so I could test and reason about each one on its own instead of debugging them tangled together.

## What's inside

**Portfolio optimization**

Markowitz mean-variance optimizer for the max Sharpe and min volatility portfolios, with the efficient frontier built by solving a constrained quadratic program at each target return. Long only, fully invested, no short selling.

**Risk engine**

Three ways of estimating VaR on the same portfolio: historical simulation (empirical, no distributional assumption), parametric (variance-covariance, closed form under normality), and Monte Carlo (10,000 correlated simulations via Cholesky decomposition). Expected Shortfall (CVaR) comes alongside every VaR estimate, since VaR alone doesn't say how bad the tail actually is.

**Data pipeline**

Price fetching via yfinance, daily returns, annualization.

**Visualizations**

Efficient frontier with the tangency and minimum-variance portfolios marked, the portfolio return distribution with VaR thresholds overlaid, a comparison of the three VaR methods across confidence levels, and how portfolio volatility responds to shocking individual weights.

## Quick start

```bash
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

```bash
pytest tests/ -v
```

Tests run on synthetic correlated returns rather than live data, so they stay fast and deterministic. They check that optimized weights sum to one, that the minimum-variance portfolio actually has lower volatility than an equal-weight benchmark, that the efficient frontier is monotonic in return, and that the three VaR methods stay within a reasonable range of each other.

portfolio_optimization/
├── base.py                    PortfolioData and Portfolio dataclasses
├── data/
│   └── market_data.py         Price fetching and returns computation
├── optimization/
│   └── markowitz.py           Mean-variance optimizer, efficient frontier
├── risk/
│   ├── historical.py          Empirical VaR
│   ├── parametric.py          Variance-covariance VaR
│   ├── monte_carlo.py         Simulated VaR (Cholesky decomposition)
│   └── engine.py              Combines all three into one summary
└── utils/
    └── visualization.py       Frontier, VaR distribution, sensitivity plots


## Math

Portfolio return and volatility, given weights $w$, expected returns $\mu$, and covariance matrix $\Sigma$:

$$R_p = w^\top \mu, \qquad \sigma_p = \sqrt{w^\top \Sigma w}$$

Max Sharpe optimization solves, subject to $\sum w_i = 1$ and $w_i \geq 0$:

$$\max_w \frac{R_p - r_f}{\sigma_p}$$

Parametric VaR assumes portfolio returns are normal, so the loss at confidence level $c$ is:

$$\text{VaR}_c = -(\mu_p + z_{1-c}\,\sigma_p), \qquad z_{1-c} = \Phi^{-1}(1-c)$$

Monte Carlo VaR draws correlated asset returns using the Cholesky factor $L$ of $\Sigma$, so that $LL^\top = \Sigma$:

$$r_{\text{sim}} = \mu + Lz, \qquad z \sim \mathcal{N}(0, I)$$

and then takes the empirical percentile of the resulting simulated portfolio returns, the same way the historical method does on real data.

## License

MIT
