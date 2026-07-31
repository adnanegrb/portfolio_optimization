"""Fetches historical prices and turns them into daily returns."""

import pandas as pd
import yfinance as yf

from portfolio_optimization.base import PortfolioData


def fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Downloads adjusted close prices for a list of tickers."""
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    prices = prices.dropna(how="all")
    return prices


def load_portfolio_data(tickers: list[str], start: str, end: str) -> PortfolioData:
    """Fetches prices and wraps the resulting daily returns in a PortfolioData."""
    prices = fetch_prices(tickers, start, end)
    returns = prices.pct_change().dropna()
    return PortfolioData(returns=returns)
