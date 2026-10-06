from pathlib import Path

import joblib

from data_preprocessing import (
    load_vendor_invoice_data,
    prepare_features,
    split_data
)

from modelling_evaluation import (
    train_linear_regression,
    train_decision_tree,
    train_random_forest,
    evaluate_model
)


def main():

    # --------------------------------------------------
    # Configuration
    # --------------------------------------------------

    db_path = Path("data/inventory.db")
    model_dir = Path("models")

    # Features used by the model
    feature_names = [
        "Quantity",
        "Dollars"
    ]

    target_name = "Freight"

    # --------------------------------------------------
    # Validate database
    # --------------------------------------------------

    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found: {db_path}"
        )

    # Create model directory if it doesn't exist
    model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 60)
    print("Vendor Freight Prediction")
    print("=" * 60)

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------

    print("\nLoading data...")

    df = load_vendor_invoice_data(
        str(db_path)
    )

    print(f"Rows loaded: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # --------------------------------------------------
    # Prepare features
    # --------------------------------------------------

    print("\nPreparing features...")

    X, y = prepare_features(df)

    print(f"Features: {feature_names}")
    print(f"Target: {target_name}")
    print(f"Valid rows: {len(X)}")

    # --------------------------------------------------
    # Train/test split
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("\nData split:")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows : {len(X_test)}")

    # --------------------------------------------------
    # Train models
    # --------------------------------------------------

    print("\nTraining models...")

    lr_model = train_linear_regression(
        X_train,
        y_train
    )

    dt_model = train_decision_tree(
        X_train,
        y_train,
        max_depth=5
    )

    rf_model = train_random_forest(
        X_train,
        y_train,
        max_depth=6,
        n_estimators=100
    )

    # --------------------------------------------------
    # Evaluate models
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    results = []

    results.append(
        evaluate_model(
            lr_model,
            X_test,
            y_test,
            "Linear Regression"
        )
    )

    results.append(
        evaluate_model(
            dt_model,
            X_test,
            y_test,
            "Decision Tree Regression"
        )
    )

    results.append(
        evaluate_model(
            rf_model,
            X_test,
            y_test,
            "Random Forest Regression"
        )
    )

    # --------------------------------------------------
    # Select best model using lowest MAE
    # --------------------------------------------------

    best_model_info = min(
        results,
        key=lambda result: result["mae"]
    )

    best_model_name = (
        best_model_info["model_name"]
    )

    models = {
        "Linear Regression": lr_model,
        "Decision Tree Regression": dt_model,
        "Random Forest Regression": rf_model
    }

    best_model = models[best_model_name]

    # --------------------------------------------------
    # Save model and metadata
    # --------------------------------------------------

    model_package = {
        "model": best_model,
        "model_name": best_model_name,
        "features": feature_names,
        "target": target_name,
        "metrics": best_model_info
    }

    model_path = (
        model_dir /
        "predict_freight_model.pkl"
    )

    joblib.dump(
        model_package,
        model_path
    )

    # --------------------------------------------------
    # Display final results
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)

    print(
        f"Selected model : {best_model_name}"
    )

    print(
        f"MAE            : "
        f"{best_model_info['mae']:.2f}"
    )

    print(
        f"RMSE           : "
        f"{best_model_info['rmse']:.2f}"
    )

    print(
        f"R²             : "
        f"{best_model_info['r2']:.2%}"
    )

    print(
        f"\nModel saved to : {model_path}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
