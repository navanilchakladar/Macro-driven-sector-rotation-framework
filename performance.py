import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calculate_metrics(returns):
    returns = returns.dropna()

    annual_return = (1 + returns).prod() ** (12 / len(returns)) - 1
    annual_vol = returns.std() * np.sqrt(12)

    sharpe = annual_return / annual_vol if annual_vol != 0 else np.nan

    cumulative = (1 + returns).cumprod()
    drawdown = cumulative / cumulative.cummax() - 1
    max_drawdown = drawdown.min()

    return {
        "Annualized Return": annual_return,
        "Annualized Volatility": annual_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_drawdown
    }


def performance_summary(backtest):

    strategy = calculate_metrics(backtest["Strategy_Return"])
    benchmark = calculate_metrics(backtest["Benchmark_Return"])

    summary = pd.DataFrame({
        "Strategy": strategy,
        "Benchmark": benchmark
    })

    return summary


def plot_cumulative_returns(backtest):

    plt.figure(figsize=(10, 6))

    plt.plot(
        backtest["Date"],
        backtest["Strategy_Cumulative"],
        label="Sector Rotation Strategy"
    )

    plt.plot(
        backtest["Date"],
        backtest["Benchmark_Cumulative"],
        label="Benchmark"
    )

    plt.title("Cumulative Returns")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.grid(True)
    plt.show()
