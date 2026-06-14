import pandas as pd
import os

DECISION_PATH = "data/decisions/squad_decisions.csv"

FORMATION = {
    "G": 1,
    "D": 4,
    "M": 3,
    "F": 3
}

def main():

    if not os.path.exists(DECISION_PATH):
        raise Exception("Decision file missing. Run decision engine first.")

    df = pd.read_csv(DECISION_PATH)

    print("\nAvailable Teams:")
    print(sorted(df["team_name"].unique()))

    team_name = input("\nEnter team name exactly as shown: ").strip()

    team_df = df[df["team_name"] == team_name].copy()

    if team_df.empty:
        raise Exception("Team not found.")

    # =====================================================
    # RANK BY OPTIMIZED ORDER
    # =====================================================

    team_df = team_df.sort_values(
        ["expected_rating", "prob_outperform", "fitness_score"],
        ascending=False
    ).reset_index(drop=True)

    # Assign squad roles
    team_df["decision"] = "BENCH"
    team_df.loc[:10, "decision"] = "START"
    team_df.loc[11:15, "decision"] = "ROTATE"

    start_df = team_df[team_df["decision"] == "START"]

    final_xi = []

    # =====================================================
    # FORMATION ENFORCEMENT
    # =====================================================

    for pos, required_count in FORMATION.items():
        pos_players = start_df[start_df["position"] == pos]
        selected = pos_players.head(required_count)
        final_xi.append(selected)

    lineup = pd.concat(final_xi)

    # Fallback if formation incomplete
    if len(lineup) < 11:
        remaining_needed = 11 - len(lineup)
        remaining_players = start_df[
            ~start_df["player_id"].isin(lineup["player_id"])
        ]
        lineup = pd.concat([lineup, remaining_players.head(remaining_needed)])

    lineup = lineup.sort_values(
        ["expected_rating", "prob_outperform", "fitness_score"],
        ascending=False
    )

    rotate_df = team_df[team_df["decision"] == "ROTATE"]
    bench_df = team_df[team_df["decision"] == "BENCH"]

    # =====================================================
    # OUTPUT
    # =====================================================

    print("\n=== STARTING XI (Optimized) ===")
    print(lineup[[
        "player_name",
        "position",
        "expected_rating",
        "prob_outperform",
        "fitness_score"
    ]])

    print("\n=== ROTATION OPTIONS ===")
    print(rotate_df[[
        "player_name",
        "position",
        "expected_rating",
        "prob_outperform",
        "fitness_score"
    ]])

    print("\n=== BENCH DEPTH ===")
    print(bench_df[[
        "player_name",
        "position",
        "expected_rating",
        "prob_outperform",
        "fitness_score"
    ]])

    print("\nCounts:")
    print("START:", len(lineup))
    print("ROTATE:", len(rotate_df))
    print("BENCH:", len(bench_df))


if __name__ == "__main__":
    main()