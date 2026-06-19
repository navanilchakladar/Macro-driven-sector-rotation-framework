from src.data_loader import load_sector_data, load_macro_data
from src.preprocessing import prepare_monthly_data
from src.models import run_walk_forward_predictions
from src.backtest import build_sector_rotation_portfolio
from src.performance import performance_summary, plot_cumulative_returns


def main():

    sector_file = "data/S&P 500 Industry returns_weights.xlsx"
    macro_folder = "data/predictors/"

    sector_returns, sector_weights = load_sector_data(sector_file)
    macro_data = load_macro_data(macro_folder)

    model_data = prepare_monthly_data(
        sector_returns=sector_returns,
        sector_weights=sector_weights,
        macro_data=macro_data
    )

    predictions = run_walk_forward_predictions(
        model_data=model_data,
        train_window=12,
        model_type="lasso"
    )

    backtest = build_sector_rotation_portfolio(
        predictions=predictions,
        sector_returns=sector_returns,
        sector_weights=sector_weights,
        top_n=5
    )

    summary = performance_summary(backtest)
    print(summary)

    plot_cumulative_returns(backtest)


if __name__ == "__main__":
    main()
