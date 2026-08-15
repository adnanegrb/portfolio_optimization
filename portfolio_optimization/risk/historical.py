import numpy as np
import pandas as pd


class HistoricalVaR:
    

    def __init__(self, returns: pd.DataFrame, weights: np.ndarray):
        self.portfolio_returns = returns @ weights

    def compute(self, confidence: float = 0.95) -> float:
        
        alpha = 1 - confidence
        return -np.percentile(self.portfolio_returns, alpha * 100)

    def expected_shortfall(self, confidence: float = 0.95) -> float:
        
        var = self.compute(confidence)
        tail_losses = self.portfolio_returns[self.portfolio_returns <= -var]
        return -tail_losses.mean() if len(tail_losses) > 0 else var
