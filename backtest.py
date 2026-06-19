import numpy as np
import pandas as pd


def build_sector_rotation_portfolio(
    predictions,
    sector_returns,
    sector_weights,
    top_n=5
):
    """
    Converts predictions into portfolio weights.

    Logic:
    - Rank sectors by predicted return
    - Select top N sectors
    - Allocate proportionally to benchmark weights
    - Compare against benchmark
    """

    df = predictions.copy()

    df["Rank"] = df.groupby("Date")["Prediction"].rank(
        ascending=False,
        method="first"
    )

    df["Selected"] = np.where(df["Rank"] <= top_n, 1, 0)

    # Placeholder benchmark weight
    # Replace this with your actual sector benchmark weight merge.
    if "Benchmark_Weight" not in df.columns:
        df["Benchmark_Weight"] = 1 / top_n

    df["Raw_Strategy_Weight"] = df["Selected"] * df["Benchmark_Weight"]

    df["Strategy_Weight"] = df.groupby("Date")["Raw_Strategy_Weight"].transform(
        lambda x: x / x.sum() if x.sum() != 0 else 0
    )

    df["Strategy_Return_Contribution"] = (
        df["Strategy_Weight"] * df["Actual_Return"]
    )

    df["Benchmark_Return_Contribution"] = (
        df["Benchmark_Weight"] * df["Actual_Return"]
    )

    backtest = df.groupby("Date").agg({
        "Strategy_Return_Contribution": "sum",
        "Benchmark_Return_Contribution": "sum"
    }).reset_index()

    backtest.rename(columns={
        "Strategy_Return_Contribution": "Strategy_Return",
        "Benchmark_Return_Contribution": "Benchmark_Return"
    }, inplace=True)

    backtest["Strategy_Cumulative"] = (
        1 + backtest["Strategy_Return"]
    ).cumprod()

    backtest["Benchmark_Cumulative"] = (
        1 + backtest["Benchmark_Return"]
    ).cumprod()

    return backtest
