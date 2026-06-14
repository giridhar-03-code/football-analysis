import requests
import pandas as pd
import time
import os
from config.config import (
    API_KEY,
    BASE_URL,
    LEAGUE_ID,
    SEASON,
    MAX_REQUESTS_PER_DAY,
    REQUEST_SLEEP_SECONDS,
    DAILY_FIXTURE_LIMIT
)

HEADERS = {
    "x-apisports-key": API_KEY
}

request_count = 0


# ====================================
# SAFE REQUEST WRAPPER
# ====================================
def safe_request(endpoint, params):
    global request_count

    if request_count >= MAX_REQUESTS_PER_DAY:
        raise Exception("Daily API request limit reached. Stopping safely.")

    response = requests.get(
        f"{BASE_URL}/{endpoint}",
        headers=HEADERS,
        params=params
    )

    request_count += 1

    if response.status_code != 200:
        raise Exception(f"Bad API response: {response.status_code}")

    data = response.json()

    if "response" not in data:
        raise Exception("Malformed API response.")

    return data["response"]


# ====================================
# GET FIXTURES
# ====================================
def get_finished_fixtures():
    fixtures = safe_request(
        "fixtures",
        {
            "league": LEAGUE_ID,
            "season": SEASON,
            "status": "FT"
        }
    )

    if not fixtures:
        raise Exception("No fixtures returned.")

    fixtures_sorted = sorted(
        fixtures,
        key=lambda x: x["fixture"]["date"],
        reverse=True
    )

    fixture_map = {}

    for f in fixtures_sorted:
        fixture_id = f["fixture"]["id"]
        date = f["fixture"]["date"]

        home_id = f["teams"]["home"]["id"]
        away_id = f["teams"]["away"]["id"]

        fixture_map[fixture_id] = {
            "date": date,
            "home_id": home_id,
            "away_id": away_id
        }

    return fixture_map


# ====================================
# GET PLAYER STATS (WITH LOGOS + PHOTOS)
# ====================================
def get_player_stats_for_fixture(fixture_id, fixture_info):

    stats = safe_request(
        "fixtures/players",
        {"fixture": fixture_id}
    )

    if not stats:
        print(f"No player data for fixture {fixture_id}")
        return []

    rows = []

    home_id = fixture_info["home_id"]
    away_id = fixture_info["away_id"]

    for team in stats:
        team_id = team["team"]["id"]
        team_name = team["team"]["name"]
        team_logo = team["team"]["logo"]

        if team_id == home_id:
            opponent_id = away_id
            home_flag = 1
        else:
            opponent_id = home_id
            home_flag = 0

        for player in team["players"]:

            statistics = player.get("statistics", [])
            if not statistics:
                continue

            games_data = statistics[0].get("games", {})

            rating = games_data.get("rating")
            minutes = games_data.get("minutes")
            position = games_data.get("position")

            player_photo = player["player"]["photo"]

            rows.append({
                "fixture_id": fixture_id,
                "team_id": team_id,
                "team_name": team_name,
                "team_logo": team_logo,
                "opponent_team_id": opponent_id,
                "home_flag": home_flag,
                "player_id": player["player"]["id"],
                "player_name": player["player"]["name"],
                "player_photo": player_photo,
                "minutes": minutes,
                "rating": rating,
                "position": position
            })

    return rows


# ====================================
# MAIN PIPELINE
# ====================================
def main():
    global request_count

    raw_path = "data/raw/player_stats_raw.csv"
    os.makedirs("data/raw", exist_ok=True)

    # --------------------------------
    # Load existing fixtures (incremental safe)
    # --------------------------------
    existing_fixtures = set()

    if os.path.exists(raw_path):
        existing_df = pd.read_csv(raw_path)
        existing_fixtures = set(existing_df["fixture_id"].unique())
        print(f"Found {len(existing_fixtures)} existing fixtures.")
    else:
        existing_df = None

    fixture_map = get_finished_fixtures()

    new_fixtures = {
        fx_id: info
        for fx_id, info in fixture_map.items()
        if fx_id not in existing_fixtures
    }

    if not new_fixtures:
        print("No new fixtures to collect.")
        return

    fixture_items = list(new_fixtures.items())[:DAILY_FIXTURE_LIMIT]

    print(f"Collecting {len(fixture_items)} fixtures.")

    all_rows = []

    for fx_id, info in fixture_items:
        print(f"Fetching fixture {fx_id}")

        rows = get_player_stats_for_fixture(fx_id, info)

        for row in rows:
            row["date"] = info["date"]

        all_rows.extend(rows)

        time.sleep(REQUEST_SLEEP_SECONDS)

    if not all_rows:
        print("No data collected.")
        return

    df_new = pd.DataFrame(all_rows)

    required_columns = [
        "fixture_id",
        "team_id",
        "team_name",
        "team_logo",
        "opponent_team_id",
        "home_flag",
        "player_id",
        "player_name",
        "player_photo",
        "minutes",
        "rating",
        "date",
        "position"
    ]

    for col in required_columns:
        if col not in df_new.columns:
            raise Exception(f"Missing column: {col}")

    if existing_df is not None:
        df_final = pd.concat([existing_df, df_new], ignore_index=True)
    else:
        df_final = df_new

    df_final = df_final.drop_duplicates(
        subset=["fixture_id", "player_id"]
    )

    df_final.to_csv(raw_path, index=False)

    print("\n=== RAW DATA VALIDATION ===")
    print("Shape:", df_final.shape)
    print("Unique Fixtures:", df_final["fixture_id"].nunique())
    print("Unique Teams:", df_final["team_id"].nunique())
    print("Unique Players:", df_final["player_id"].nunique())
    print("API Requests Used:", request_count)


if __name__ == "__main__":
    main()