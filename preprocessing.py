import pandas as pd


def month_start_date(df, date_col="Date"):
    df[date_col] = pd.to_datetime(df[date_col])
    df[date_col] = (
        df[date_col].dt.floor("D")
        + pd.offsets.MonthEnd(0)
        - pd.offsets.MonthBegin(1)
    )
    return df


def prepare_monthly_data(sector_returns, sector_weights, macro_data):
    """
    Prepares final modeling dataset.

    Main idea:
    - Convert all dates to monthly frequency
    - Lag macro variables to avoid look-ahead bias
    - Merge sector returns with predictors
    """

    macro_data = month_start_date(macro_data, "Date")
    macro_data = macro_data.sort_values("Date")

    macro_cols = [col for col in macro_data.columns if col != "Date"]

    # Lag macro variables by one month
    macro_data[macro_cols] = macro_data[macro_cols].shift(1)

    # Placeholder structure:
    # You can later replace this with your melt logic from Spyder.
    model_data = {
        "sector_returns": sector_returns,
        "sector_weights": sector_weights,
        "macro_data": macro_data
    }

    return model_data
