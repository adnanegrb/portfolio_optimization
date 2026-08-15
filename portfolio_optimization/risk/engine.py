import numpy as np
import pandas as pd

from portfolio_optimization.base import PortfolioData
from portfolio_optimization.risk.historical import HistoricalVaR
from portfolio_optimization.risk.parametric import ParametricVaR
from portfolio_optimization.risk.monte_carlo import MonteCarloVaR


class RiskEngine:
    

    def __init__(self, data: PortfolioData, weights: np.ndarray, n_simulations: int = 10_000):
        self.historical = HistoricalVaR(data.returns, weights)
        self.parametric = ParametricVaR(data.mean_returns, data.cov_matrix, weights)
        self.monte_carlo = MonteCarloVaR(data.mean_returns, data.cov_matrix, weights, n_simulations)

    def summary(self, confidence_levels: tuple[float, ...] = (0.95, 0.99)) -> pd.DataFrame:
        
        rows = []
        methods = {
            "Historical": self.historical,
            "Parametric": self.parametric,
            "Monte Carlo": self.monte_carlo,
        }
        for name, method in methods.items():
            for cl in confidence_levels:
                rows.append({
                    "method": name,
                    "confidence": cl,
                    "VaR": method.compute(cl),
                    "ES": method.expected_shortfall(cl),
                })
        return pd.DataFrame(rows)
