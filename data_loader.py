import os
import pandas as pd


def load_sector_data(file_path):
    """
    Loads sector returns and benchmark weights from Excel.

    Expected format:
    - First few columns identify sector / industry
    - Return columns are monthly dates
    - Weight columns are monthly benchmark weights
    """

    raw = pd.read_excel(file_path)

    raw = raw.dropna(how="all")
    raw = raw.reset_index(drop=True)

    sector_returns = raw.copy()
    sector_weights = raw.copy()

    return sector_returns, sector_weights


def load_macro_data(folder_path):
    """
    Loads macro predictor CSV files from a folder.

    Each CSV should contain:
    - Date column
    - One value column
    """

    macro_frames = []

    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            path = os.path.join(folder_path, file)

            temp = pd.read_csv(path)

            temp.columns = [col.strip() for col in temp.columns]

            if "Date" not in temp.columns:
                if "DATE" in temp.columns:
                    temp.rename(columns={"DATE": "Date"}, inplace=True)

            value_cols = [col for col in temp.columns if col != "Date"]

            if len(value_cols) > 0:
                var_name = file.replace(".csv", "").replace(" ", "_")
                temp = temp[["Date", value_cols[0]]]
                temp.rename(columns={value_cols[0]: var_name}, inplace=True)

                macro_frames.append(temp)

    macro_data = macro_frames[0]

    for frame in macro_frames[1:]:
        macro_data = pd.merge(macro_data, frame, on="Date", how="outer")

    return macro_data
