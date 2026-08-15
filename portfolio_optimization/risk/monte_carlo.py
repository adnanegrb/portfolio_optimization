import numpy as np


class MonteCarloVaR:
    
    def __init__(self, mean_returns, cov_matrix, weights: np.ndarray, n_simulations: int = 10_000, seed: int = 42):
        self.mean_returns = mean_returns.values / 252
        self.cov_matrix = cov_matrix.values / 252
        self.weights = weights
        self.n_simulations = n_simulations
        self.rng = np.random.default_rng(seed)
        self.simulated_returns = self._simulate()

    def _simulate(self) -> np.ndarray:
        chol = np.linalg.cholesky(self.cov_matrix)
        z = self.rng.standard_normal((self.n_simulations, len(self.weights)))
        asset_returns = self.mean_returns + z @ chol.T
        return asset_returns @ self.weights

    def compute(self, confidence: float = 0.95) -> float:
        alpha = 1 - confidence
        return -np.percentile(self.simulated_returns, alpha * 100)

    def expected_shortfall(self, confidence: float = 0.95) -> float:
        var = self.compute(confidence)
        tail_losses = self.simulated_returns[self.simulated_returns <= -var]
        return -tail_losses.mean() if len(tail_losses) > 0 else var
