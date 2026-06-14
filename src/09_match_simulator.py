import pandas as pd
import os
import joblib

FEATURE_PATH = "data/features/matchfit_features.csv"
CLASSIFIER_PATH = "data/models/performance_classifier.pkl"
REGRESSOR_PATH = "data/models/performance_regressor.pkl"

def main():

    if not os.path.exists(FEATURE_PATH):
        raise Exception("Feature file missing.")

    df = pd.read_csv(FEATURE_PATH)
    df["date"] = pd.to_datetime(df["date"])

    classifier = joblib.load(CLASSIFIER_PATH)
    regressor = joblib.load(REGRESSOR_PATH)

    print("\nAvailable Teams:")
    teams = sorted(df["team_name"].unique())
    print(teams)

    team_name = input("\nEnter YOUR team: ").strip()
    opponent_name = input("Enter OPPONENT team: ").strip()
    venue = input("Home or Away? (H/A): ").strip().upper()

    if team_name not in teams or opponent_name not in teams:
        raise Exception("Invalid team selection.")

    if venue not in ["H", "A"]:
        raise Exception("Venue must be H or A.")

    # -----------------------------------
    # Latest records
    # -----------------------------------
    latest_df = (
        df.sort_values("date")
        .groupby("player_id")
        .tail(1)
        .reset_index(drop=True)
    )

    team_df = latest_df[latest_df["team_name"] == team_name].copy()
    opponent_df = latest_df[latest_df["team_name"] == opponent_name]

    if team_df.empty or opponent_df.empty:
        raise Exception("Insufficient data.")

    # -----------------------------------
    # Compute opponent strength
    # -----------------------------------
    opponent_strength = opponent_df["rolling_rating_5"].mean()
    league_mean = latest_df["rolling_rating_5"].mean()

    strength_gap = opponent_strength - league_mean

    print("\nOpponent Strength Index:", round(strength_gap, 4))

    # Inject context
    team_df["opponent_rolling_avg_rating"] = opponent_strength
    team_df["home_flag"] = 1 if venue == "H" else 0

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

    X_sim = team_df[feature_cols]

    # -----------------------------------
    # Run models
    # -----------------------------------
    team_df["prob_outperform"] = classifier.predict_proba(X_sim)[:, 1]
    team_df["predicted_delta"] = regressor.predict(X_sim)

    team_df["fitness_score"] = (1 - team_df["fatigue_ratio"]).clip(0, 1)

    # -----------------------------------
    #  Player-Specific Tactical Adjustment
    # -----------------------------------
    TACTICAL_MULTIPLIER = 0.8  # adjust 0.6–1.0 depending on intensity

    vulnerability = (
        team_df["rolling_rating_std_5"].fillna(0)
        + (1 - team_df["fitness_score"])
    )

    player_adjustment = strength_gap * TACTICAL_MULTIPLIER * vulnerability

    team_df["expected_rating"] = (
        team_df["rolling_rating_5"]
        + team_df["predicted_delta"]
        - player_adjustment
    ).clip(0, 10)

    # -----------------------------------
    # Sort for projection
    # -----------------------------------
    team_df = team_df.sort_values(
        ["expected_rating", "prob_outperform", "fitness_score"],
        ascending=False
    ).reset_index(drop=True)

    print(f"\n=== SIMULATED MATCH: {team_name} vs {opponent_name} ===")
    print("(Home)" if venue == "H" else "(Away)")
    print("\nTop 15 Projected Players:\n")

    print(team_df[[
        "player_name",
        "expected_rating",
        "prob_outperform",
        "fitness_score"
    ]].head(15))


if __name__ == "__main__":
    main()