import sqlite3

import pandas as pd
from sklearn.model_selection import train_test_split


def load_vendor_invoice_data(db_path: str) -> pd.DataFrame:
    """
    Load vendor invoice data from SQLite database.
    """
    query = "SELECT * FROM vendor_invoice"

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn)

    return df


def prepare_features(df: pd.DataFrame):
    """
    Select features and target variable.
    """
    required_columns = ["Quantity", "Dollars", "Freight"]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    X = df[["Quantity", "Dollars"]]
    y = df["Freight"]

    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Split dataset into train and test sets.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )
