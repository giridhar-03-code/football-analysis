import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

FEATURE_PATH = "data/features/matchfit_features.csv"
MODEL_PATH = "data/models/performance_regressor.pkl"

def main():

    if not os.path.exists(FEATURE_PATH):
        raise Exception("Feature file missing. Run feature engineering first.")

    df = pd.read_csv(FEATURE_PATH)

    print("\n=== REGRESSION TRAINING START ===")
    print("Dataset Shape:", df.shape)

    if "target_delta" not in df.columns:
        raise Exception("target_delta missing. Check feature engineering.")

    feature_cols = [
        "rolling_rating_3",
        "rolling_rating_5",
        "rolling_minutes_3",
        "rolling_minutes_5",
        "rolling_rating_std_5",
        "days_since_last_match",
        "fatigue_ratio",
        "form_delta",
        "match_count",
        "home_flag",
        "opponent_rolling_avg_rating"
    ]

    for col in feature_cols:
        if col not in df.columns:
            raise Exception(f"Missing feature column: {col}")

    X = df[feature_cols]
    y = df["target_delta"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Train Size:", len(X_train))
    print("Test Size:", len(X_test))

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    print("\n=== REGRESSION PERFORMANCE ===")
    print("Train R2:", round(r2_score(y_train, train_pred), 4))
    print("Test R2:", round(r2_score(y_test, test_pred), 4))
    print("Train MAE:", round(mean_absolute_error(y_train, train_pred), 4))
    print("Test MAE:", round(mean_absolute_error(y_test, test_pred), 4))

    importances = pd.Series(
        model.feature_importances_,
        index=feature_cols
    ).sort_values(ascending=False)

    print("\n=== FEATURE IMPORTANCE ===")
    print(importances)

    os.makedirs("data/models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("\nModel saved to:", MODEL_PATH)


if __name__ == "__main__":
    main()