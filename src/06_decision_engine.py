import pandas as pd
import os
import joblib

FEATURE_PATH = "data/features/matchfit_features.csv"
CLASSIFIER_PATH = "data/models/performance_classifier.pkl"
REGRESSOR_PATH = "data/models/performance_regressor.pkl"
OUTPUT_PATH = "data/decisions/squad_decisions.csv"

def main():

    if not os.path.exists(FEATURE_PATH):
        raise Exception("Feature file missing.")

    if not os.path.exists(CLASSIFIER_PATH):
        raise Exception("Classifier model missing.")

    if not os.path.exists(REGRESSOR_PATH):
        raise Exception("Regressor model missing.")

    df = pd.read_csv(FEATURE_PATH)

    print("\n=== DECISION ENGINE START ===")
    print("Feature Dataset Shape:", df.shape)

    classifier = joblib.load(CLASSIFIER_PATH)
    regressor = joblib.load(REGRESSOR_PATH)

    df["date"] = pd.to_datetime(df["date"])

    # ---------------------------------
    # Get latest record per player
    # ---------------------------------
    latest_df = (
        df.sort_values("date")
        .groupby("player_id")
        .tail(1)
        .reset_index(drop=True)
    )

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
        if col not in latest_df.columns:
            raise Exception(f"Missing feature column: {col}")

    X_latest = latest_df[feature_cols]

    # ---------------------------------
    # Classification: probability
    # ---------------------------------
    probs = classifier.predict_proba(X_latest)[:, 1]
    latest_df["prob_outperform"] = probs

    # ---------------------------------
    # Regression: predicted delta
    # ---------------------------------
    predicted_delta = regressor.predict(X_latest)
    latest_df["predicted_delta"] = predicted_delta

    # Compute expected rating
    latest_df["expected_rating"] = (
        latest_df["rolling_rating_5"] + latest_df["predicted_delta"]
    )

    # Clip rating to realistic bounds
    latest_df["expected_rating"] = latest_df["expected_rating"].clip(0, 10)

    # ---------------------------------
    # Fitness Score
    # ---------------------------------
    latest_df["fitness_score"] = 1 - latest_df["fatigue_ratio"]
    latest_df["fitness_score"] = latest_df["fitness_score"].clip(0, 1)

    # No tactical label here
    latest_df["decision"] = "UNASSIGNED"

    os.makedirs("data/decisions", exist_ok=True)
    latest_df.to_csv(OUTPUT_PATH, index=False)

    print("\n=== DECISION ENGINE COMPLETE ===")
    print("Players Evaluated:", len(latest_df))
    print("Output saved to:", OUTPUT_PATH)

    print("\nPreview:")
    print(latest_df[[
        "player_name",
        "prob_outperform",
        "expected_rating",
        "fitness_score"
    ]].head())


if __name__ == "__main__":
    main()