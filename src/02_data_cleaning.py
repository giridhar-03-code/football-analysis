import pandas as pd
import os

RAW_PATH = "data/raw/player_stats_raw.csv"
CLEAN_PATH = "data/cleaned/player_stats_cleaned.csv"

def main():
    if not os.path.exists(RAW_PATH):
        raise Exception("Raw file does not exist. Run data collection first.")

    df = pd.read_csv(RAW_PATH)

    print("\n=== BEFORE CLEANING ===")
    print("Shape:", df.shape)

    # Drop rows with missing rating
    df = df.dropna(subset=["rating"])

    # Convert rating to float
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    # Drop invalid ratings
    df = df.dropna(subset=["rating"])

    # Remove zero-minute players
    df = df[df["minutes"] > 0]

    # Drop duplicates
    df = df.drop_duplicates(subset=["fixture_id", "player_id"])

    if df.empty:
        raise Exception("Cleaning resulted in empty dataset.")

    os.makedirs("data/cleaned", exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)

    print("\n=== AFTER CLEANING ===")
    print("Shape:", df.shape)
    print("Unique Players:", df["player_id"].nunique())
    print("Unique Teams:", df["team_id"].nunique())


if __name__ == "__main__":
    main()