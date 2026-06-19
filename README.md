# Macro-Driven Sector Rotation Framework

This project builds a macro-driven sector rotation model for equity industry allocation.

The model forecasts next-month sector returns using macroeconomic variables such as inflation, dollar index, industrial production, credit spreads, treasury slope, and commodity prices.

## Objective

The objective is to test whether macro variables can improve sector allocation decisions relative to a benchmark equity portfolio.

## Methodology

1. Load sector returns and benchmark weights.
2. Load macro predictor variables.
3. Convert all datasets to monthly frequency.
4. Lag macro variables by one month to avoid look-ahead bias.
5. Forecast next-month sector returns using walk-forward models.
6. Select sectors with the strongest predicted returns.
7. Reweight selected sectors proportionally.
8. Compare strategy performance against the benchmark.

## Models

The framework supports:

- Linear Regression
- LASSO Regression
- Random Forest Regression

## Portfolio Construction

Each month:

- Forecast returns for each sector.
- Rank sectors by predicted return.
- Select the top N sectors.
- Allocate capital across selected sectors.
- Compare against benchmark returns.

## Performance Metrics

The backtest reports:

- Annualized return
- Annualized volatility
- Sharpe ratio
- Maximum drawdown
- Cumulative return

## Project Structure

```text
Macro-driven-sector-rotation-framework/
│
├── main.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── variable_selection.py
│   ├── models.py
│   ├── backtest.py
│   └── performance.py
│
└── data/
    ├── predictors/
    └── S&P 500 Industry returns_weights.xlsx
