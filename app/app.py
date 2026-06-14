import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from html import escape

st.set_page_config(layout="wide", page_title="MatchFit Tactical Engine")

FEATURE_PATH = "data/features/matchfit_features.csv"
CLASSIFIER_PATH = "data/models/performance_classifier.pkl"
REGRESSOR_PATH = "data/models/performance_regressor.pkl"

# =====================================================
# LOAD
# =====================================================

@st.cache_data
def load_data():
    df = pd.read_csv(FEATURE_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_resource
def load_models():
    clf = joblib.load(CLASSIFIER_PATH)
    reg = joblib.load(REGRESSOR_PATH)
    return clf, reg


df = load_data()
classifier, regressor = load_models()

latest_df = (
    df.sort_values("date")
    .groupby("player_id")
    .tail(1)
    .reset_index(drop=True)
)

teams = sorted(latest_df["team_name"].unique())

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("MatchFit Tactical Engine")

team_name = st.sidebar.selectbox("Your Team", teams)
opponent_name = st.sidebar.selectbox("Opponent", teams)
venue = st.sidebar.radio("Venue", ["Home", "Away"])

formations = {
    "4-3-3": {"G": 1, "D": 4, "M": 3, "F": 3},
    "4-2-3-1": {"G": 1, "D": 4, "M": 5, "F": 1},
    "3-5-2": {"G": 1, "D": 3, "M": 5, "F": 2},
}

formation_choice = st.sidebar.selectbox(
    "Formation",
    list(formations.keys())
)

tactical_multiplier = st.sidebar.slider(
    "Tactical Intensity",
    0.4,
    1.5,
    0.9,
    0.1,
)

# =====================================================
# SIMULATION
# =====================================================

team_df = latest_df[latest_df["team_name"] == team_name].copy()
opponent_df = latest_df[latest_df["team_name"] == opponent_name].copy()

opponent_strength = opponent_df["rolling_rating_5"].mean()
league_mean = latest_df["rolling_rating_5"].mean()

strength_gap = opponent_strength - league_mean

team_df["opponent_rolling_avg_rating"] = opponent_strength
team_df["home_flag"] = 1 if venue == "Home" else 0

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
    "opponent_rolling_avg_rating",
]

X_sim = team_df[feature_cols]

team_df["prob_outperform"] = classifier.predict_proba(X_sim)[:, 1]
team_df["predicted_delta"] = regressor.predict(X_sim)
team_df["fitness_score"] = (1 - team_df["fatigue_ratio"]).clip(0, 1)

vulnerability = (
    team_df["rolling_rating_std_5"].fillna(0)
    + (1 - team_df["fitness_score"])
)
team_df["risk_factor"] = vulnerability.clip(lower=0)

player_adjustment = strength_gap * tactical_multiplier * team_df["risk_factor"]

team_df["expected_rating"] = (
    team_df["rolling_rating_5"]
    + team_df["predicted_delta"]
    - player_adjustment
).clip(0, 10)

team_df = team_df.sort_values(
    ["expected_rating", "prob_outperform", "fitness_score"],
    ascending=False,
).reset_index(drop=True)

# =====================================================
# FORMATION SELECTION
# =====================================================

formation = formations[formation_choice]
lineup_blocks = []

for pos, req in formation.items():
    pos_players = team_df[team_df["position"] == pos]
    lineup_blocks.append(pos_players.head(req))

starting_xi = pd.concat(lineup_blocks)

if len(starting_xi) < 11:
    remaining = team_df[
        ~team_df["player_id"].isin(starting_xi["player_id"])
    ]
    starting_xi = pd.concat([starting_xi, remaining.head(11 - len(starting_xi))])

starting_xi = starting_xi.sort_values(
    ["expected_rating"],
    ascending=False,
).reset_index(drop=True)

# =====================================================
# WIN PROBABILITY
# =====================================================

team_strength = starting_xi["expected_rating"].mean()
opponent_strength_live = opponent_df["rolling_rating_5"].mean()

strength_diff = team_strength - opponent_strength_live

win_prob = 1 / (1 + np.exp(-1.5 * strength_diff))
draw_prob = max(0.05, 0.2 * (1 - abs(strength_diff)))
loss_prob = 1 - win_prob - draw_prob

# =====================================================
# HEADER METRICS
# =====================================================

st.markdown(
    f"""
    <div class="hero-shell">
        <div class="hero-left">
            <div class="hero-kicker">MATCHFIT // TACTICAL SIM</div>
            <div class="hero-title">{escape(team_name)} <span>VS</span> {escape(opponent_name)}</div>
            <div class="hero-subtitle">Venue: {escape(venue)} | Formation: {escape(formation_choice)}</div>
        </div>
        <div class="hero-right">
            <div class="hero-chip">Intensity {tactical_multiplier:.1f}</div>
            <div class="hero-chip">XI Strength {team_strength:.2f}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)
c1.metric("Win %", round(win_prob, 2))
c2.metric("Draw %", round(draw_prob, 2))
c3.metric("Loss %", round(loss_prob, 2))

st.progress(min(team_strength / 10, 1.0))
st.caption("Projected Team Strength")

# =====================================================
# CONFIDENCE TIERS
# =====================================================


def confidence(rank, fitness):
    if rank <= 2:
        base = "HIGH"
    elif rank <= 6:
        base = "MEDIUM"
    else:
        base = "RISKY"
    if fitness < 0.5:
        if base == "HIGH":
            return "MEDIUM"
        if base == "MEDIUM":
            return "RISKY"
    return base


starting_xi["confidence"] = [
    confidence(i, starting_xi.loc[i, "fitness_score"])
    for i in range(len(starting_xi))
]

# =====================================================
# CSS THEME
# =====================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&family=Rajdhani:wght@400;600;700&display=swap');

:root {
    --bg-0: #070a12;
    --bg-1: #0f1324;
    --bg-2: #171c34;
    --text: #f6f8ff;
    --text-dim: #a8b0cd;
    --orange: #f43c04;
    --cyan: #00e5ff;
    --lime: #a7ff4f;
    --danger: #ff5472;
    --line: rgba(255, 255, 255, 0.14);
}

@keyframes pulseGlow {
    0% { box-shadow: 0 0 0 rgba(244, 60, 4, 0.0), 0 0 0 rgba(0, 229, 255, 0.0); }
    50% { box-shadow: 0 0 24px rgba(244, 60, 4, 0.28), 0 0 34px rgba(0, 229, 255, 0.18); }
    100% { box-shadow: 0 0 0 rgba(244, 60, 4, 0.0), 0 0 0 rgba(0, 229, 255, 0.0); }
}

@keyframes scanLine {
    0% { transform: translateY(-120%); opacity: 0; }
    10% { opacity: 0.45; }
    90% { opacity: 0.18; }
    100% { transform: translateY(220%); opacity: 0; }
}

@keyframes borderRun {
    0% { background-position: 0% 50%; }
    100% { background-position: 200% 50%; }
}

@keyframes chipEnter {
    0% { opacity: 0; transform: translateY(18px) scale(0.96); }
    100% { opacity: 1; transform: translateY(0) scale(1); }
}

@keyframes holoSweep {
    0% { transform: translateX(-120%); opacity: 0; }
    20% { opacity: 0.2; }
    100% { transform: translateX(120%); opacity: 0; }
}

* {
    letter-spacing: 0.01em;
}

html, body, [class*="css"] {
    font-family: "Rajdhani", "Segoe UI", sans-serif;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(1100px 700px at 85% -10%, rgba(244, 60, 4, 0.22) 0%, rgba(244, 60, 4, 0) 55%),
        radial-gradient(900px 650px at 10% -20%, rgba(0, 229, 255, 0.2) 0%, rgba(0, 229, 255, 0) 56%),
        linear-gradient(155deg, #060912 0%, #0a1020 40%, #111832 100%);
    color: var(--text);
}

[data-testid="stSidebar"] {
    background:
        radial-gradient(350px 180px at 80% -20%, rgba(244, 60, 4, 0.28) 0%, rgba(244, 60, 4, 0) 60%),
        linear-gradient(175deg, #0a0f1d 0%, #0f1427 100%);
    border-right: 1px solid rgba(0, 229, 255, 0.28);
}

[data-testid="stSidebar"] * {
    color: var(--text) !important;
}

[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [role="slider"] {
    background: linear-gradient(145deg, var(--orange), #ff7d52);
    border: 2px solid #ffd4c6;
    box-shadow: 0 0 12px rgba(244, 60, 4, 0.5);
}

[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stRadio label {
    color: var(--text-dim) !important;
    text-transform: uppercase;
    font-size: 0.76rem;
    letter-spacing: 0.08em;
}

[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stRadio > div {
    background: rgba(6, 10, 18, 0.8);
    border: 1px solid rgba(0, 229, 255, 0.28);
    border-radius: 10px;
}

.hero-shell {
    position: relative;
    display: flex;
    justify-content: space-between;
    gap: 14px;
    align-items: flex-start;
    background:
        linear-gradient(130deg, rgba(15, 22, 42, 0.9) 0%, rgba(20, 29, 55, 0.9) 100%);
    border: 1px solid rgba(0, 229, 255, 0.4);
    border-radius: 16px;
    padding: 16px 18px;
    margin-bottom: 12px;
    overflow: hidden;
    animation: pulseGlow 4s ease-in-out infinite;
}

.hero-shell::before {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(90deg, rgba(244, 60, 4, 0.26), rgba(0, 229, 255, 0.22), rgba(244, 60, 4, 0.26));
    background-size: 200% 100%;
    animation: borderRun 8s linear infinite;
    mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
    mask-composite: exclude;
    padding: 1px;
    pointer-events: none;
}

.hero-title {
    font-family: "Orbitron", "Rajdhani", sans-serif;
    font-size: 1.9rem;
    line-height: 1.2;
    color: #ffffff;
    font-weight: 700;
    text-transform: uppercase;
    margin-top: 3px;
}

.hero-subtitle {
    color: var(--text-dim);
    font-size: 0.9rem;
    margin-top: 5px;
}

.hero-title span {
    color: var(--orange);
    text-shadow: 0 0 20px rgba(244, 60, 4, 0.58);
}

.hero-kicker {
    font-family: "Orbitron", "Rajdhani", sans-serif;
    color: var(--cyan);
    text-transform: uppercase;
    font-size: 0.74rem;
    letter-spacing: 0.14em;
    font-weight: 700;
}

.hero-right {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    justify-content: flex-end;
    align-self: center;
}

.hero-chip {
    background: rgba(6, 12, 22, 0.78);
    border: 1px solid rgba(244, 60, 4, 0.45);
    color: #ffd8cb;
    border-radius: 999px;
    padding: 5px 10px;
    font-size: 0.78rem;
    font-weight: 600;
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(11, 17, 33, 0.92), rgba(17, 24, 46, 0.88));
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 8px 10px;
    box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.04);
}

[data-testid="stMetricLabel"] {
    color: var(--text-dim);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    font-size: 0.7rem !important;
}

[data-testid="stMetricValue"] {
    color: #fff9f7;
    text-shadow: 0 0 12px rgba(244, 60, 4, 0.25);
}

[data-testid="stProgressBar"] > div > div > div {
    background: linear-gradient(90deg, var(--cyan) 0%, var(--orange) 100%);
}

.section-title {
    font-family: "Orbitron", "Rajdhani", sans-serif;
    color: var(--text);
    font-weight: 700;
    font-size: 1.02rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 12px 0 10px 0;
}

.pitch-shell {
    position: relative;
    border: 1px solid rgba(0, 229, 255, 0.42);
    border-radius: 22px;
    padding: 14px;
    background:
        radial-gradient(140% 85% at 50% -20%, rgba(0, 229, 255, 0.22) 0%, rgba(0, 229, 255, 0) 52%),
        radial-gradient(120% 75% at 95% 120%, rgba(244, 60, 4, 0.2) 0%, rgba(244, 60, 4, 0) 54%),
        linear-gradient(180deg, #0c2e24 0%, #0a241c 42%, #071612 100%);
    overflow: hidden;
    box-shadow:
        0 26px 40px rgba(0, 0, 0, 0.45),
        0 0 24px rgba(0, 229, 255, 0.1),
        inset 0 0 0 1px rgba(255, 255, 255, 0.06);
}

.pitch-shell::before {
    content: "";
    position: absolute;
    inset: 0;
    background:
        repeating-linear-gradient(
            180deg,
            rgba(107, 206, 138, 0.17) 0 36px,
            rgba(77, 173, 108, 0.24) 36px 72px
        );
    mix-blend-mode: soft-light;
    opacity: 0.95;
    pointer-events: none;
}

.pitch-shell::after {
    content: "";
    position: absolute;
    inset: -8px;
    background: linear-gradient(120deg, rgba(255, 255, 255, 0), rgba(255, 255, 255, 0.2), rgba(255, 255, 255, 0));
    animation: holoSweep 5.5s ease-in-out infinite;
    pointer-events: none;
}

.pitch-overlay {
    position: absolute;
    inset: 12px;
    border: 2px solid rgba(234, 255, 246, 0.62);
    border-radius: 16px;
    pointer-events: none;
    z-index: 1;
}

.pitch-overlay::before {
    content: "";
    position: absolute;
    left: 50%;
    top: 50%;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: rgba(234, 255, 246, 0.8);
    transform: translate(-50%, -50%);
}

.pitch-halfway {
    position: absolute;
    left: 0;
    top: 50%;
    width: 100%;
    border-top: 2px solid rgba(234, 255, 246, 0.62);
    transform: translateY(-50%);
}

.pitch-center-circle {
    position: absolute;
    left: 50%;
    top: 50%;
    width: 110px;
    height: 110px;
    border: 2px solid rgba(234, 255, 246, 0.62);
    border-radius: 50%;
    transform: translate(-50%, -50%);
}

.pitch-box {
    position: absolute;
    left: 50%;
    width: min(42%, 260px);
    height: 86px;
    border: 2px solid rgba(234, 255, 246, 0.6);
    transform: translateX(-50%);
}

.pitch-box.top {
    top: 0;
    border-top: none;
    border-radius: 0 0 12px 12px;
}

.pitch-box.bottom {
    bottom: 0;
    border-bottom: none;
    border-radius: 12px 12px 0 0;
}

.pitch-head {
    position: relative;
    z-index: 3;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    margin: 2px 4px 12px 4px;
}

.pitch-title {
    font-family: "Orbitron", "Rajdhani", sans-serif;
    color: #ecfff7;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: 0.78rem;
}

.pitch-meta {
    color: #c5f7ea;
    font-size: 0.73rem;
    border: 1px solid rgba(0, 229, 255, 0.38);
    background: rgba(7, 26, 21, 0.64);
    border-radius: 999px;
    padding: 3px 10px;
}

.pitch-content {
    position: relative;
    z-index: 2;
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.pitch-row {
    display: grid;
    grid-template-columns: 122px 1fr;
    align-items: center;
    gap: 12px;
    position: relative;
    padding: 4px 2px;
}

.row-label {
    position: relative;
    text-align: left;
    width: fit-content;
    padding: 5px 18px 5px 10px;
    border-radius: 8px;
    font-family: "Orbitron", "Rajdhani", sans-serif;
    font-size: 0.66rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #e8fff5;
    border: 1px solid rgba(235, 255, 245, 0.62);
    background: rgba(7, 25, 17, 0.66);
}

.row-label::after {
    content: "";
    position: absolute;
    right: -12px;
    top: 50%;
    transform: translateY(-50%);
    width: 0;
    height: 0;
    border-top: 9px solid transparent;
    border-bottom: 9px solid transparent;
    border-left: 12px solid rgba(7, 25, 17, 0.66);
}

.pitch-row.line-f .row-label {
    border-color: rgba(244, 60, 4, 0.72);
    color: #ffd9ce;
}

.pitch-row.line-m .row-label {
    border-color: rgba(0, 229, 255, 0.72);
    color: #cff8ff;
}

.pitch-row.line-x .row-label {
    border-color: rgba(244, 60, 4, 0.72);
    color: #ffd7c9;
}

.pitch-row.line-d .row-label,
.pitch-row.line-g .row-label {
    border-color: rgba(167, 255, 79, 0.72);
    color: #e7ffd0;
}

.row-players {
    display: flex;
    justify-content: center;
    gap: 12px;
    flex-wrap: wrap;
    min-height: 102px;
}

.player-chip {
    position: relative;
    width: 224px;
    background: linear-gradient(145deg, rgba(6, 16, 29, 0.92), rgba(10, 29, 48, 0.9));
    border: 1px solid rgba(151, 232, 255, 0.24);
    border-left: 4px solid rgba(0, 229, 255, 0.82);
    border-radius: 14px;
    padding: 9px 10px 10px 10px;
    backdrop-filter: blur(2px);
    transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
    opacity: 0;
    animation: chipEnter 0.45s ease forwards;
    animation-delay: var(--delay, 0s);
}

.player-chip::before {
    content: "";
    position: absolute;
    inset: 0;
    border-radius: 14px;
    background: linear-gradient(160deg, rgba(255, 255, 255, 0.13), rgba(255, 255, 255, 0));
    pointer-events: none;
}

.player-chip::after {
    content: "";
    position: absolute;
    right: 0;
    top: 0;
    width: 26px;
    height: 26px;
    clip-path: polygon(100% 0, 0 0, 100% 100%);
    background: rgba(255, 255, 255, 0.16);
    border-top-right-radius: 14px;
    pointer-events: none;
}

.player-chip:hover {
    transform: translateY(-3px);
}

.player-chip.high {
    border-left-color: var(--lime);
    border-color: rgba(167, 255, 79, 0.32);
    box-shadow: 0 0 0 1px rgba(167, 255, 79, 0.42), 0 0 22px rgba(167, 255, 79, 0.16);
}

.player-chip.medium {
    border-left-color: var(--cyan);
    border-color: rgba(0, 229, 255, 0.36);
    box-shadow: 0 0 0 1px rgba(0, 229, 255, 0.34), 0 0 20px rgba(0, 229, 255, 0.16);
}

.player-chip.risky {
    border-left-color: var(--danger);
    border-color: rgba(255, 84, 114, 0.35);
    box-shadow: 0 0 0 1px rgba(255, 84, 114, 0.36), 0 0 20px rgba(255, 84, 114, 0.16);
}

.chip-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 8px;
}

.chip-index {
    color: #8ec8ff;
    font-size: 0.62rem;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    margin-bottom: 2px;
}

.chip-pos {
    font-size: 0.64rem;
    color: #052239;
    font-weight: 800;
    background: linear-gradient(145deg, #d5f4ff, #95dfff);
    border-radius: 999px;
    min-width: 28px;
    text-align: center;
    padding: 4px 8px;
}

.player-name {
    color: #f2fbff;
    font-size: 0.92rem;
    font-weight: 700;
    line-height: 1.12;
}

.chip-badge {
    margin: 7px 0 8px 0;
    display: inline-flex;
    align-items: center;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 0.62rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.chip-badge.high {
    color: #b9ff5f;
    background: rgba(167, 255, 79, 0.14);
    border: 1px solid rgba(167, 255, 79, 0.36);
}

.chip-badge.medium {
    color: #9ff2ff;
    background: rgba(0, 229, 255, 0.14);
    border: 1px solid rgba(0, 229, 255, 0.34);
}

.chip-badge.risky {
    color: #ffb1c0;
    background: rgba(255, 84, 114, 0.15);
    border: 1px solid rgba(255, 84, 114, 0.34);
}

.chip-stats {
    display: grid;
    gap: 5px;
}

.stat-row {
    background: rgba(8, 26, 43, 0.52);
    border: 1px solid rgba(191, 233, 255, 0.16);
    border-radius: 8px;
    padding: 3px 6px 5px 6px;
}

.stat-head {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    line-height: 1;
}

.stat-head span {
    font-size: 0.58rem;
    letter-spacing: 0.08em;
    color: #8ab6d1;
}

.stat-head b {
    color: #edf8ff;
    font-size: 0.72rem;
}

.stat-bar {
    height: 4px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.12);
    margin-top: 4px;
    overflow: hidden;
}

.stat-bar i {
    display: block;
    height: 100%;
    width: var(--w, 50%);
    border-radius: inherit;
}

.stat-row.exp .stat-bar i {
    background: linear-gradient(90deg, #23d8ff 0%, #13f7be 100%);
}

.stat-row.out .stat-bar i {
    background: linear-gradient(90deg, #76f78f 0%, #e9ff7a 100%);
}

.stat-row.delta .stat-bar i {
    background: linear-gradient(90deg, #8f9eff 0%, #d5a8ff 100%);
}

.stat-row.risk .stat-bar i {
    background: linear-gradient(90deg, #ff944b 0%, #ff5472 100%);
}

.focus-card {
    background: linear-gradient(145deg, rgba(12, 19, 38, 0.9) 0%, rgba(22, 32, 58, 0.86) 100%);
    border: 1px solid rgba(0, 229, 255, 0.35);
    border-left: 6px solid var(--orange);
    border-radius: 12px;
    padding: 12px;
    box-shadow: 0 0 24px rgba(0, 229, 255, 0.1);
}

.focus-name {
    color: #ffffff;
    font-weight: 700;
    font-size: 1.15rem;
    margin-bottom: 8px;
    text-transform: uppercase;
}

.focus-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 8px;
}

.focus-pill {
    background: rgba(8, 13, 27, 0.62);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 8px;
    padding: 7px 8px;
    color: #cfd8f4;
    font-size: 0.8rem;
}

.focus-pill b {
    color: #9ff2ff;
}

[data-testid="stDataFrame"] {
    border: 1px solid rgba(0, 229, 255, 0.25);
    border-radius: 10px;
    overflow: hidden;
}

[data-testid="stCaptionContainer"] {
    color: #96a0c5 !important;
}

[data-testid="stMarkdownContainer"] p {
    color: var(--text);
}

@media (max-width: 900px) {
    .hero-shell {
        flex-direction: column;
        align-items: flex-start;
    }
    .hero-right {
        justify-content: flex-start;
    }
    .pitch-head {
        flex-direction: column;
        align-items: flex-start;
        gap: 6px;
    }
    .pitch-row {
        grid-template-columns: 1fr;
        gap: 6px;
    }
    .row-label {
        width: fit-content;
        margin: 0 auto;
    }
    .row-label::after {
        display: none;
    }
    .player-chip {
        width: min(100%, 360px);
    }
}
</style>
""",
    unsafe_allow_html=True,
)

# =====================================================
# PITCH DISPLAY
# =====================================================

st.markdown('<div class="section-title">Starting XI Tactical Board</div>', unsafe_allow_html=True)

position_rows = [
    ("line-f", "Attack", starting_xi[starting_xi["position"] == "F"]),
    ("line-m", "Midfield", starting_xi[starting_xi["position"] == "M"]),
    ("line-d", "Defence", starting_xi[starting_xi["position"] == "D"]),
    ("line-g", "Goalkeeper", starting_xi[starting_xi["position"] == "G"]),
]

placed_ids = set()
for _, _, row_df in position_rows:
    placed_ids.update(row_df["player_id"].tolist())

extras = starting_xi[~starting_xi["player_id"].isin(placed_ids)]
if not extras.empty:
    position_rows.insert(1, ("line-x", "Utility", extras))

pitch_html = [
    '<div class="pitch-shell">',
    '<div class="pitch-overlay"><div class="pitch-halfway"></div><div class="pitch-center-circle"></div><div class="pitch-box top"></div><div class="pitch-box bottom"></div></div>',
    f'<div class="pitch-head"><div class="pitch-title">Formation {escape(formation_choice)} Tactical Grid</div><div class="pitch-meta">XI Strength {team_strength:.2f}</div></div>',
    '<div class="pitch-content">',
]
chip_order = 0
for row_class, row_label, row_df in position_rows:
    if row_df.empty:
        continue
    row_html = [f'<div class="pitch-row {row_class}"><div class="row-label">{escape(row_label)}</div><div class="row-players">']
    for _, player in row_df.iterrows():
        chip_order += 1
        conf_value = str(player.get("confidence", "MEDIUM")).upper()
        conf_class = conf_value.lower()
        if conf_class not in {"high", "medium", "risky"}:
            conf_class = "medium"
        exp_rating = float(np.nan_to_num(player["expected_rating"], nan=0.0))
        outperform = float(np.nan_to_num(player["prob_outperform"], nan=0.0))
        delta = float(np.nan_to_num(player["predicted_delta"], nan=0.0))
        risk = float(np.nan_to_num(player["risk_factor"], nan=0.0))

        exp_w = int(np.clip((exp_rating / 10.0) * 100.0, 0, 100))
        out_w = int(np.clip(outperform * 100.0, 0, 100))
        delta_w = int(np.clip((np.tanh(delta) + 1.0) * 50.0, 0, 100))
        risk_w = int(np.clip((risk / 2.5) * 100.0, 0, 100))
        row_html.append(
            f'<article class="player-chip {conf_class}" style="--delay:{chip_order * 0.04:.2f}s">'
            '<div class="chip-top"><div>'
            f'<div class="chip-index">#{chip_order:02d}</div>'
            f'<div class="player-name">{escape(str(player["player_name"]))}</div></div>'
            f'<div class="chip-pos">{escape(str(player["position"]))}</div></div>'
            f'<div class="chip-badge {conf_class}">{escape(conf_value)} confidence</div>'
            '<div class="chip-stats">'
            f'<div class="stat-row exp"><div class="stat-head"><span>EXP</span><b>{exp_rating:.2f}</b></div><div class="stat-bar"><i style="--w:{exp_w}%"></i></div></div>'
            f'<div class="stat-row out"><div class="stat-head"><span>OUT</span><b>{outperform:.2f}</b></div><div class="stat-bar"><i style="--w:{out_w}%"></i></div></div>'
            f'<div class="stat-row delta"><div class="stat-head"><span>DELTA</span><b>{delta:+.2f}</b></div><div class="stat-bar"><i style="--w:{delta_w}%"></i></div></div>'
            f'<div class="stat-row risk"><div class="stat-head"><span>RISK</span><b>{risk:.2f}</b></div><div class="stat-bar"><i style="--w:{risk_w}%"></i></div></div>'
            '</div>'
            '</article>'
        )
    row_html.append('</div></div>')
    pitch_html.append("".join(row_html))
pitch_html.append('</div></div>')

st.markdown("".join(pitch_html), unsafe_allow_html=True)

player_details = starting_xi[
    [
        "player_name",
        "position",
        "expected_rating",
        "predicted_delta",
        "prob_outperform",
        "risk_factor",
        "fitness_score",
        "confidence",
    ]
].copy()
player_details = player_details.rename(
    columns={
        "player_name": "Player",
        "position": "Pos",
        "expected_rating": "Expected Rating",
        "predicted_delta": "Return Delta",
        "prob_outperform": "Outperform Prob",
        "risk_factor": "Risk Factor",
        "fitness_score": "Fitness",
        "confidence": "Confidence",
    }
)
st.dataframe(
    player_details.round(
        {
            "Expected Rating": 2,
            "Return Delta": 2,
            "Outperform Prob": 2,
            "Risk Factor": 2,
            "Fitness": 2,
        }
    ),
    hide_index=True,
    use_container_width=True,
)

# =====================================================
# PLAYER CARD
# =====================================================

st.markdown('<div class="section-title">Player Focus Card</div>', unsafe_allow_html=True)

player_options = starting_xi["player_id"].tolist()
if player_options:
    if "selected" not in st.session_state or st.session_state["selected"] not in player_options:
        st.session_state["selected"] = player_options[0]

    id_to_label = {
        pid: f"{name} ({pos})"
        for pid, name, pos in zip(
            starting_xi["player_id"], starting_xi["player_name"], starting_xi["position"]
        )
    }

    st.selectbox(
        "Select Player",
        player_options,
        format_func=lambda pid: id_to_label.get(pid, str(pid)),
        key="selected",
    )

if "selected" in st.session_state:
    p = starting_xi[starting_xi["player_id"] == st.session_state["selected"]].iloc[0]
    left, right = st.columns([1.25, 1], gap="large")

    with left:
        st.markdown(
            f"""
            <div class="focus-card">
                <div class="focus-name">{escape(p["player_name"])} | {p["position"]}</div>
                <div class="focus-grid">
                    <div class="focus-pill"><b>Expected Rating:</b> {p["expected_rating"]:.2f}</div>
                    <div class="focus-pill"><b>Outperform Prob:</b> {p["prob_outperform"]:.2f}</div>
                    <div class="focus-pill"><b>Return Delta:</b> {p["predicted_delta"]:+.2f}</div>
                    <div class="focus-pill"><b>Risk Factor:</b> {p["risk_factor"]:.2f}</div>
                    <div class="focus-pill"><b>Fitness Score:</b> {p["fitness_score"]:.2f}</div>
                    <div class="focus-pill"><b>Confidence Tier:</b> {p["confidence"]}</div>
                    <div class="focus-pill"><b>Rolling Form (5):</b> {p["rolling_rating_5"]:.2f}</div>
                    <div class="focus-pill"><b>Consistency:</b> {p["rolling_rating_std_5"]:.2f}</div>
                    <div class="focus-pill"><b>Form Delta:</b> {p["form_delta"]:+.2f}</div>
                    <div class="focus-pill"><b>Days Since Match:</b> {int(p["days_since_last_match"])}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(float(np.clip(p["fitness_score"], 0, 1)))

    with right:
        radar_vals = [
            p["rolling_rating_5"],
            p["fitness_score"] * 10,
            max(0, 10 - p["rolling_rating_std_5"] * 2),
            p["form_delta"] + 5,
            p["expected_rating"],
        ]
        categories = ["Form", "Fitness", "Consistency", "Momentum", "Expected"]

        fig = go.Figure()
        fig.add_trace(
            go.Scatterpolar(
                r=radar_vals,
                theta=categories,
                fill="toself",
                line=dict(color="#00e5ff", width=2.4),
                fillcolor="rgba(0,229,255,0.22)",
            )
        )
        fig.update_layout(
            polar=dict(
                bgcolor="rgba(10,15,30,0.85)",
                radialaxis=dict(
                    range=[0, 10],
                    gridcolor="rgba(0,229,255,0.22)",
                    tickcolor="#00e5ff",
                    tickfont=dict(color="#d7e8ff"),
                ),
                angularaxis=dict(gridcolor="rgba(244,60,4,0.24)", tickfont=dict(color="#f6f8ff")),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e9f1ff"),
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

# =====================================================
# SUBSTITUTION SUGGESTIONS
# =====================================================

bench = team_df.iloc[11:]
subs = []

for _, starter in starting_xi.iterrows():
    same_pos = bench[bench["position"] == starter["position"]]
    if same_pos.empty:
        continue
    best = same_pos.sort_values(
        ["fitness_score", "prob_outperform"],
        ascending=False,
    ).iloc[0]
    if best["fitness_score"] > starter["fitness_score"] + 0.2:
        subs.append({
            "OUT": starter["player_name"],
            "IN": best["player_name"],
            "Reason": "Higher Fitness",
        })

if subs:
    st.subheader("Suggested Substitutions")
    st.dataframe(pd.DataFrame(subs), hide_index=True, use_container_width=True)
