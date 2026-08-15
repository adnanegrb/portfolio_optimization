import numpy as np
from scipy.stats import norm


class ParametricVaR:

    def __init__(self, mean_returns, cov_matrix, weights: np.ndarray):
        self.weights = weights
        self.portfolio_mean = float(weights @ mean_returns) / 252
        self.portfolio_vol = float(np.sqrt(weights @ cov_matrix @ weights)) / np.sqrt(252)

    def compute(self, confidence: float = 0.95) -> float:
        z = norm.ppf(1 - confidence)
        return -(self.portfolio_mean + z * self.portfolio_vol)

    def expected_shortfall(self, confidence: float = 0.95) -> float:
        z = norm.ppf(1 - confidence)
        es_multiplier = norm.pdf(z) / (1 - confidence)
        return -(self.portfolio_mean - self.portfolio_vol * es_multiplier)
