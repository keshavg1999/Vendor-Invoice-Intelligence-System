import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def train_linear_regression(X_train, y_train):
    """
    Train a Linear Regression model.
    """
    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )

    return model


def train_decision_tree(
    X_train,
    y_train,
    max_depth=5
):
    """
    Train a Decision Tree Regression model.
    """
    model = DecisionTreeRegressor(
        max_depth=max_depth,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    return model


def train_random_forest(
    X_train,
    y_train,
    max_depth=6,
    n_estimators=100
):
    """
    Train a Random Forest Regression model.
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    return model


def evaluate_model(
    model,
    X_test,
    y_test,
    model_name
):
    """
    Evaluate a regression model using MAE, RMSE and R².
    """

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        predictions
    )

    print(f"\n{model_name} Performance")
    print("-" * 40)
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.2%}")

    return {
        "model_name": model_name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }
