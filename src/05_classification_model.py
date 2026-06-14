import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

FEATURE_PATH = "data/features/matchfit_features.csv"
MODEL_PATH = "data/models/performance_classifier.pkl"

def main():

    if not os.path.exists(FEATURE_PATH):
        raise Exception("Feature file missing.")

    df = pd.read_csv(FEATURE_PATH)

    print("\n=== CLASSIFICATION TRAINING START ===")
    print("Dataset Shape:", df.shape)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

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

    X = df[feature_cols]
    y = df["target_outperform"]

    # Time-aware split
    split_index = int(len(df) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print("Train Size:", len(X_train))
    print("Test Size:", len(X_test))

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=4,
        min_samples_leaf=10,
        random_state=42
    )

    model.fit(X_train, y_train)

    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)

    train_f1 = f1_score(y_train, train_preds)
    test_f1 = f1_score(y_test, test_preds)

    print("\n=== CLASSIFICATION PERFORMANCE ===")
    print("Train Accuracy:", round(train_acc, 4))
    print("Test Accuracy:", round(test_acc, 4))
    print("Train F1:", round(train_f1, 4))
    print("Test F1:", round(test_f1, 4))

    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, test_preds))

    os.makedirs("data/models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("\nModel saved to:", MODEL_PATH)


if __name__ == "__main__":
    main()