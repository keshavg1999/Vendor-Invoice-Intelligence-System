import joblib
import pandas as pd


MODEL_PATH = "models/predict_freight_model.pkl"


def load_model(model_path: str = MODEL_PATH):
    """
    Load the trained freight prediction model package.
    """

    package = joblib.load(model_path)

    if not isinstance(package, dict):
        raise ValueError(
            "The saved model file does not contain "
            "the expected model package."
        )

    if "model" not in package:
        raise ValueError(
            "The saved model package does not contain a model."
        )

    return package


def predict_freight_cost(input_data):
    """
    Predict freight cost for new vendor invoices.

    Parameters
    ----------
    input_data : dict
        Dictionary containing:
        - Quantity
        - Dollars

    Returns
    -------
    pd.DataFrame
        Input data with predicted freight cost.
    """

    # Load model package
    package = load_model()

    model = package["model"]
    features = package["features"]

    # Convert input dictionary to DataFrame
    input_df = pd.DataFrame(input_data)

    # Validate required features
    missing_features = [
        feature
        for feature in features
        if feature not in input_df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    # Keep features in the same order used during training
    X_new = input_df[features]

    # Validate numeric values
    for feature in features:
        X_new[feature] = pd.to_numeric(
            X_new[feature],
            errors="coerce"
        )

    if X_new.isnull().any().any():
        raise ValueError(
            "Input contains missing or non-numeric values."
        )

    # Make predictions
    predictions = model.predict(X_new)

    # Add predictions to output
    result = input_df.copy()

    result["Predicted_Freight"] = predictions.round(2)

    return result


if __name__ == "__main__":

    # -----------------------------------------------
    # Example inference run
    # -----------------------------------------------

    sample_data = {
        "Quantity": [100, 200, 300, 400],
        "Dollars": [18500, 9000, 3000, 200]
    }

    prediction = predict_freight_cost(
        sample_data
    )

    print("\nFreight Predictions")
    print("-" * 40)

    print(prediction.to_string(index=False))
