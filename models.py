import pandas as pd
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def get_model(model_type="lasso"):

    if model_type == "linear":
        return LinearRegression()

    if model_type == "lasso":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LassoCV(cv=5, random_state=42))
        ])

    if model_type == "random_forest":
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=4,
            random_state=42
        )

    raise ValueError("model_type must be linear, lasso, or random_forest")


def run_single_sector_model(df, sector, train_window=12, model_type="lasso"):
    """
    Walk-forward prediction for one sector.

    Required columns:
    - Date
    - Sector
    - Return
    - macro predictors
    """

    df = df[df["Sector"] == sector].copy()
    df = df.sort_values("Date")

    results = []

    feature_cols = [
        col for col in df.columns
        if col not in ["Date", "Sector", "Return"]
    ]

    for i in range(train_window, len(df)):

        train = df.iloc[i - train_window:i]
        test = df.iloc[[i]]

        X_train = train[feature_cols]
        y_train = train["Return"]

        X_test = test[feature_cols]

        model = get_model(model_type)
        model.fit(X_train, y_train)

        pred = model.predict(X_test)[0]

        results.append({
            "Date": test["Date"].iloc[0],
            "Sector": sector,
            "Actual_Return": test["Return"].iloc[0],
            "Prediction": pred
        })

    return pd.DataFrame(results)


def run_walk_forward_predictions(model_data, train_window=12, model_type="lasso"):
    """
    Placeholder function.

    After your data is converted into long format, this will run
    sector-by-sector walk-forward predictions.
    """

    df = model_data.get("model_frame")

    if df is None:
        raise ValueError(
            "model_data must contain a long-format dataframe called 'model_frame'."
        )

    all_results = []

    for sector in df["Sector"].unique():
        result = run_single_sector_model(
            df=df,
            sector=sector,
            train_window=train_window,
            model_type=model_type
        )
        all_results.append(result)

    return pd.concat(all_results, ignore_index=True)
