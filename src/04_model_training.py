import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

FEATURE_PATH = "data/features/matchfit_features.csv"
MODEL_PATH = "data/models/rating_model.pkl"

def main():

    if not os.path.exists(FEATURE_PATH):
        raise Exception("Feature file missing. Run feature engineering first.")

    df = pd.read_csv(FEATURE_PATH)

    print("\n=== MODEL TRAINING START ===")
    print("Dataset Shape:", df.shape)

    # Ensure date exists
    if "date" not in df.columns:
        raise Exception("Date column missing in feature dataset.")

    df["date"] = pd.to_datetime(df["date"])

    # Sort chronologically (global time ordering)
    df = df.sort_values("date").reset_index(drop=True)

    # Feature columns
    feature_cols = [
         "rolling_rating_3",
    "rolling_minutes_3",
    "rolling_rating_5",
    "rolling_minutes_5",
    "rolling_rating_std_5",
    "days_since_last_match",
    "fatigue_ratio",
    "form_delta",
    "match_count"
    ]

    for col in feature_cols:
        if col not in df.columns:
            raise Exception(f"Missing feature column: {col}")

    X = df[feature_cols]
    y = df["target_next_rating"]

    
    # TIME-AWARE SPLIT (80 / 20)
    
    split_index = int(len(df) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print("Train Size:", X_train.shape[0])
    print("Test Size:", X_test.shape[0])

    # ----------------------------
    # MODEL
    # ----------------------------
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Predictions
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    # Metrics
    train_r2 = r2_score(y_train, train_preds)
    test_r2 = r2_score(y_test, test_preds)

    train_mae = mean_absolute_error(y_train, train_preds)
    test_mae = mean_absolute_error(y_test, test_preds)

    print("\n=== MODEL PERFORMANCE ===")
    print("Train R2:", round(train_r2, 4))
    print("Test R2:", round(test_r2, 4))
    print("Train MAE:", round(train_mae, 4))
    print("Test MAE:", round(test_mae, 4))

    # Feature importance
    importance = pd.Series(
        model.feature_importances_,
        index=feature_cols
    ).sort_values(ascending=False)

    print("\n=== FEATURE IMPORTANCE ===")
    print(importance)

    # Save model
    os.makedirs("data/models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("\nModel saved to:", MODEL_PATH)


if __name__ == "__main__":
    main()