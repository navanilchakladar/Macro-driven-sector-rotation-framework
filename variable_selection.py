import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def select_variables_lasso(X, y, cv=5):
    """
    LASSO variable selection.

    Keeps the project essence:
    macro variables are selected dynamically based on their usefulness
    in explaining next-month sector returns.
    """

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", LassoCV(cv=cv, random_state=42))
    ])

    model.fit(X, y)

    coefs = model.named_steps["lasso"].coef_

    importance = pd.DataFrame({
        "Variable": X.columns,
        "Coefficient": coefs
    })

    importance["Abs_Coefficient"] = importance["Coefficient"].abs()

    importance = importance.sort_values(
        "Abs_Coefficient",
        ascending=False
    )

    selected = importance.loc[
        importance["Coefficient"] != 0,
        "Variable"
    ].tolist()

    return selected, importance
