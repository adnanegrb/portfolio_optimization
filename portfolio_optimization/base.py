from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class PortfolioData:
    

    returns: pd.DataFrame
    tickers: list[str] = field(init=False)
    mean_returns: pd.Series = field(init=False)
    cov_matrix: pd.DataFrame = field(init=False)

    def __post_init__(self):
        self.tickers = list(self.returns.columns)
        self.mean_returns = self.returns.mean() * 252
        self.cov_matrix = self.returns.cov() * 252

    def n_assets(self) -> int:
        return len(self.tickers)


@dataclass
class Portfolio:
    

    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float

    def as_dict(self, tickers: list[str]) -> dict:
        return {ticker: round(w, 4) for ticker, w in zip(tickers, self.weights)}
