"""Plots for the efficient frontier, VaR distributions, and sensitivity analysis."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid")


def plot_efficient_frontier(frontier, max_sharpe, min_vol, tickers, save_path=None):
    """Risk/return scatter of the frontier, with the tangency and min-vol portfolios marked."""
    vols = [p.volatility for p in frontier]
    rets = [p.expected_return for p in frontier]
    sharpes = [p.sharpe_ratio for p in frontier]

    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(vols, rets, c=sharpes, cmap="viridis", s=25)
    ax.scatter(max_sharpe.volatility, max_sharpe.expected_return, c="red", marker="*", s=300, label="Max Sharpe")
    ax.scatter(min_vol.volatility, min_vol.expected_return, c="blue", marker="X", s=200, label="Min Volatility")

    fig.colorbar(scatter, label="Sharpe ratio")
    ax.set_xlabel("Annualized volatility")
    ax.set_ylabel("Annualized return")
    ax.set_title("Efficient frontier")
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_var_distribution(portfolio_returns, var_95, var_99, save_path=None):
    """Histogram of portfolio returns with the 95% and 99% VaR thresholds marked."""
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(portfolio_returns, bins=60, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(-var_95, color="orange", linestyle="--", label=f"95% VaR = {var_95:.2%}")
    ax.axvline(-var_99, color="red", linestyle="--", label=f"99% VaR = {var_99:.2%}")

    ax.set_xlabel("Daily portfolio return")
    ax.set_ylabel("Frequency")
    ax.set_title("Portfolio return distribution with VaR thresholds")
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_var_by_method(summary_df, save_path=None):
    """Bar chart comparing VaR across the three methods, at both confidence levels."""
    fig, ax = plt.subplots(figsize=(9, 6))
    pivot = summary_df.pivot(index="method", columns="confidence", values="VaR")
    pivot.plot(kind="bar", ax=ax, color=["steelblue", "indianred"])

    ax.set_ylabel("VaR (daily loss fraction)")
    ax.set_title("VaR by method and confidence level")
    ax.legend(title="Confidence", labels=["95%", "99%"])
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_weight_sensitivity(data, base_weights, tickers, shock_range=(-0.1, 0.1), save_path=None):
    """Shows how portfolio volatility reacts to shocking each asset weight independently."""
    fig, ax = plt.subplots(figsize=(9, 6))
    shocks = np.linspace(shock_range[0], shock_range[1], 15)

    for i, ticker in enumerate(tickers):
        vols = []
        for shock in shocks:
            w = base_weights.copy()
            w[i] += shock
            w = np.clip(w, 0, None)
            w = w / w.sum()
            vol = np.sqrt(w @ data.cov_matrix @ w)
            vols.append(vol)
        ax.plot(shocks, vols, label=ticker, marker="o", markersize=3)

    ax.set_xlabel("Weight shock")
    ax.set_ylabel("Portfolio volatility")
    ax.set_title("Sensitivity of volatility to individual weight shocks")
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
