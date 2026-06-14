import pandas as pd
import os

CLEAN_PATH = "data/cleaned/player_stats_cleaned.csv"
FEATURE_PATH = "data/features/matchfit_features.csv"

def main():

    if not os.path.exists(CLEAN_PATH):
        raise Exception("Cleaned file missing. Run cleaning stage first.")

    df = pd.read_csv(CLEAN_PATH)

    print("\n=== BEFORE FEATURE ENGINEERING ===")
    print("Shape:", df.shape)

    if "date" not in df.columns:
        raise Exception("Date column missing.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if df["date"].isna().any():
        raise Exception("Date conversion failed.")

    # Sort per player chronologically
    df = df.sort_values(["player_id", "date"]).reset_index(drop=True)

    # ====================================================
    # PLAYER ROLLING FEATURES
    # ====================================================

    df["rolling_rating_3"] = (
        df.groupby("player_id")["rating"]
        .rolling(3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["rolling_minutes_3"] = (
        df.groupby("player_id")["minutes"]
        .rolling(3, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )

    df["rolling_rating_5"] = (
        df.groupby("player_id")["rating"]
        .rolling(5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["rolling_minutes_5"] = (
        df.groupby("player_id")["minutes"]
        .rolling(5, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )

    df["rolling_rating_std_5"] = (
        df.groupby("player_id")["rating"]
        .rolling(5, min_periods=1)
        .std()
        .reset_index(level=0, drop=True)
    )

    df["days_since_last_match"] = (
        df.groupby("player_id")["date"]
        .diff()
        .dt.days
        .fillna(7)
    )

    df["fatigue_ratio"] = df["rolling_minutes_5"] / 450
    df["form_delta"] = df["rating"] - df["rolling_rating_5"]
    df["match_count"] = df.groupby("player_id").cumcount() + 1

    
    # OPPONENT CONTEXT FEATURES
   

    # Team average rating per fixture
    team_fixture_avg = (
        df.groupby(["fixture_id", "team_id"])["rating"]
        .mean()
        .reset_index()
        .rename(columns={"rating": "team_fixture_avg_rating"})
    )

    df = df.merge(
        team_fixture_avg,
        on=["fixture_id", "team_id"],
        how="left"
    )

    # Opponent average rating per fixture
    opponent_fixture_avg = (
        df.groupby(["fixture_id", "opponent_team_id"])["rating"]
        .mean()
        .reset_index()
        .rename(columns={
            "opponent_team_id": "team_id",
            "rating": "opponent_fixture_avg_rating"
        })
    )

    df = df.merge(
        opponent_fixture_avg,
        on=["fixture_id", "team_id"],
        how="left"
    )

    # Rolling opponent strength
    df = df.sort_values("date")

    df["opponent_rolling_avg_rating"] = (
        df.groupby("opponent_team_id")["opponent_fixture_avg_rating"]
        .rolling(5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # ====================================================
    # TARGETS
    # ====================================================

    df["target_next_rating"] = (
        df.groupby("player_id")["rating"].shift(-1)
    )

    df = df.dropna(subset=["target_next_rating"])

    #  Delta Target for Regression
    df["target_delta"] = (
        df["target_next_rating"] - df["rolling_rating_5"]
    )

    # Classification target
    df["target_outperform"] = (
        df["target_next_rating"] > df["rolling_rating_5"]
    ).astype(int)

    if df.empty:
        raise Exception("Feature dataset empty after target creation.")

    os.makedirs("data/features", exist_ok=True)
    df.to_csv(FEATURE_PATH, index=False)

    print("\n=== AFTER FEATURE ENGINEERING ===")
    print("Shape:", df.shape)
    print("Unique Players:", df["player_id"].nunique())
    print("Unique Teams:", df["team_id"].nunique())
    print("Positive Class Ratio:", df["target_outperform"].mean())


if __name__ == "__main__":
    main()