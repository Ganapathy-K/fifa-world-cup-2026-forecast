"""
Notebook 09 (script form) - Confederation calibration fix.

Problem (caught by the eye test): the Elo Monte Carlo (nb 08) floats non-European
teams (Morocco, Ecuador, Japan, Mexico, Colombia) above traditional powers
(England, Portugal). Root cause: Elo is zero-sum *within* a confederation, but the
six confederations barely play each other outside World Cups, so each pool's rating
level can drift from the global mean. Elo treats a CONMEBOL/CAF/AFC win as worth the
same as a UEFA win.

Fix: from the inter-confederation matches that DO exist in the data, fit one Elo
offset per confederation so that cross-confederation results become unbiased, then
apply those offsets to the World Cup field and re-run the same 10k simulation.
The data corrects the data - the eye test only said where to look.

Run: .venv/Scripts/python.exe fifa_wc_2026_poisson/09_confederation_adjustment.py
"""

import pandas as pd
import numpy as np
from scipy.stats import poisson
from functools import lru_cache
from pathlib import Path

from match_engine import (win_probability as engine_win_probability,
                          build_scoreline_sampler, sample_scorelines,
                          build_round_of_32)

# ----------------------------------------------------------------------------
# Config (mirrors nb 08)
# ----------------------------------------------------------------------------
PROCESSED_DATA_DIR = Path(r"D:/Data Science/Visual Studio Code/fifa_wc_2026_poisson/data/processed")
PLAYED_MATCHES_PATH = PROCESSED_DATA_DIR / "played_matches.parquet"
GROUPS_PATH = PROCESSED_DATA_DIR / "wc_groups.parquet"
GROUP_FIXTURES_PATH = PROCESSED_DATA_DIR / "wc_group_fixtures.parquet"
CHAMPION_ODDS_ELO_PATH = PROCESSED_DATA_DIR / "champion_odds_elo.parquet"
CHAMPION_ODDS_ADJ_PATH = PROCESSED_DATA_DIR / "champion_odds_elo_adjusted.parquet"
CONFED_OFFSETS_PATH = PROCESSED_DATA_DIR / "confederation_offsets.parquet"
REPORT_PATH = Path(r"D:/Data Science/Visual Studio Code/fifa_wc_2026_poisson/CHAMPION_ODDS_ADJUSTED.md")

INITIAL_RATING = 1500.0
K_FACTOR = 30.0
HOME_ADVANTAGE_ELO = 65.0
MAX_GOALS = 10
RANDOM_SEED = 2026
N_SIMULATIONS = 10000

# Confederation fit knobs
CONFED_FIT_FROM_YEAR = 2010   # recent inter-confederation form only
CONFED_FIT_EPOCHS = 300
CONFED_FIT_LR = 6.0

# ----------------------------------------------------------------------------
# Confederation map (teams with >=100 matches; all 48 WC teams covered)
# ----------------------------------------------------------------------------
CONFEDERATIONS = {
    "UEFA": [
        "Sweden", "England", "Germany", "Hungary", "Poland", "Italy", "Switzerland",
        "Netherlands", "Denmark", "Norway", "Austria", "Belgium", "Scotland", "Finland",
        "France", "Spain", "Romania", "Russia", "Bulgaria", "Wales", "Northern Ireland",
        "Portugal", "Turkey", "Republic of Ireland", "Greece", "Estonia", "Iceland",
        "Czechoslovakia", "Israel", "Yugoslavia", "Luxembourg", "Latvia", "Malta",
        "Lithuania", "Cyprus", "Albania", "Croatia", "Czech Republic", "Serbia",
        "Ukraine", "Slovakia", "Slovenia", "Belarus", "Azerbaijan", "Georgia",
        "North Macedonia", "German DR", "Moldova", "Faroe Islands",
        "Bosnia and Herzegovina", "Armenia", "Montenegro", "Kazakhstan", "Liechtenstein",
        "Andorra", "San Marino", "Gibraltar", "Kosovo",
    ],
    "CONMEBOL": [
        "Argentina", "Brazil", "Uruguay", "Chile", "Paraguay", "Peru", "Colombia",
        "Ecuador", "Bolivia", "Venezuela",
    ],
    "CONCACAF": [
        "Mexico", "United States", "Trinidad and Tobago", "Costa Rica", "Jamaica",
        "Honduras", "El Salvador", "Guatemala", "Panama", "Haiti", "Cuba", "Canada",
        "Suriname", "Curaçao", "Guyana", "Barbados", "Martinique", "Guadeloupe",
        "Grenada", "Saint Vincent and the Grenadines", "Antigua and Barbuda",
        "Saint Lucia", "Saint Kitts and Nevis", "Nicaragua", "Dominica", "Bermuda",
        "French Guiana", "Puerto Rico", "Dominican Republic", "Aruba", "Belize",
        "Cayman Islands", "British Virgin Islands",
    ],
    "CAF": [
        "Egypt", "Zambia", "Kenya", "Uganda", "Tunisia", "Ghana", "Nigeria", "Senegal",
        "Algeria", "Cameroon", "Morocco", "Tanzania", "Mali", "Guinea", "DR Congo",
        "Ivory Coast",
        "Malawi", "Zimbabwe", "South Africa", "Sudan", "Burkina Faso", "Togo", "Angola",
        "Ethiopia", "Congo", "Gabon", "Libya", "Mozambique", "Botswana", "Mauritius",
        "Benin", "Madagascar", "Liberia", "Sierra Leone", "Lesotho", "Rwanda", "Namibia",
        "Eswatini", "Mauritania", "Niger", "Gambia", "Cape Verde", "Burundi", "Zanzibar",
        "Guinea-Bissau", "Equatorial Guinea", "Seychelles", "Comoros", "Chad",
        "Central African Republic", "Somalia", "Réunion", "Djibouti",
    ],
    "AFC": [
        "South Korea", "Thailand", "Malaysia", "Japan", "Saudi Arabia", "Indonesia",
        "China PR", "Singapore", "Kuwait", "Iraq", "Qatar", "United Arab Emirates",
        "Iran", "Bahrain", "India", "Oman", "Myanmar", "Hong Kong", "Syria", "Jordan",
        "North Korea", "Uzbekistan", "Philippines", "Vietnam", "Lebanon", "Yemen",
        "Bangladesh", "Vietnam Republic", "Nepal", "Cambodia", "Sri Lanka", "Pakistan",
        "Palestine", "Taiwan", "Laos", "Maldives", "Tajikistan", "Kyrgyzstan",
        "Turkmenistan", "Afghanistan", "Brunei", "Macau", "Guam", "Mongolia", "Bhutan",
        "Australia",
    ],
    "OFC": [
        "New Zealand", "Fiji", "New Caledonia", "Tahiti", "Solomon Islands", "Vanuatu",
        "Papua New Guinea",
    ],
}
team_confederation = {team: confed for confed, teams in CONFEDERATIONS.items() for team in teams}

# ----------------------------------------------------------------------------
# 1. Rebuild Elo on the full history (identical recipe to nb 08)
# ----------------------------------------------------------------------------
played_matches = pd.read_parquet(PLAYED_MATCHES_PATH).sort_values("date").reset_index(drop=True)


def margin_multiplier(goal_difference):
    margin = abs(goal_difference)
    if margin <= 1:
        return 1.0
    if margin == 2:
        return 1.5
    return (11 + margin) / 8


ratings = {}
gap_history, goal_diff_history = [], []
for match in played_matches.itertuples(index=False):
    home_rating = ratings.get(match.home_team, INITIAL_RATING)
    away_rating = ratings.get(match.away_team, INITIAL_RATING)
    home_bonus = 0.0 if match.neutral else HOME_ADVANTAGE_ELO
    rating_gap = home_rating + home_bonus - away_rating
    expected_home = 1.0 / (1.0 + 10.0 ** (-rating_gap / 400.0))

    goal_diff = match.home_score - match.away_score
    actual_home = 1.0 if goal_diff > 0 else (0.5 if goal_diff == 0 else 0.0)
    gap_history.append(rating_gap)
    goal_diff_history.append(goal_diff)

    change = K_FACTOR * margin_multiplier(goal_diff) * (actual_home - expected_home)
    ratings[match.home_team] = home_rating + change
    ratings[match.away_team] = away_rating - change

slope, intercept = np.polyfit(gap_history, goal_diff_history, 1)
total_goals = (played_matches["home_score"] + played_matches["away_score"]).mean()
print(f"Rated {len(ratings)} teams through {played_matches['date'].max().date()}")
print(f"Supremacy fit: goal_diff = {slope:.5f} * gap + {intercept:.3f} | avg total goals {total_goals:.3f}")

# ----------------------------------------------------------------------------
# 2. Fit per-confederation Elo offsets from inter-confederation matches
# ----------------------------------------------------------------------------
fit = played_matches.copy()
fit["home_conf"] = fit["home_team"].map(team_confederation)
fit["away_conf"] = fit["away_team"].map(team_confederation)
fit = fit[(fit["date"].dt.year >= CONFED_FIT_FROM_YEAR)
          & fit["home_conf"].notna() & fit["away_conf"].notna()
          & (fit["home_conf"] != fit["away_conf"])].copy()
fit["actual_home"] = np.where(fit["home_score"] > fit["away_score"], 1.0,
                              np.where(fit["home_score"] == fit["away_score"], 0.5, 0.0))
print(f"\nInter-confederation matches since {CONFED_FIT_FROM_YEAR}: {len(fit)}")

home_R = fit["home_team"].map(ratings).to_numpy()
away_R = fit["away_team"].map(ratings).to_numpy()
home_bonus = np.where(fit["neutral"].to_numpy(), 0.0, HOME_ADVANTAGE_ELO)
actual = fit["actual_home"].to_numpy()
hc = fit["home_conf"].to_numpy()
ac = fit["away_conf"].to_numpy()

confeds = sorted(CONFEDERATIONS.keys())
offset = {c: 0.0 for c in confeds}
for _ in range(CONFED_FIT_EPOCHS):
    oh = np.array([offset[c] for c in hc])
    oa = np.array([offset[c] for c in ac])
    gap = (home_R + oh + home_bonus) - (away_R + oa)
    pred = 1.0 / (1.0 + 10.0 ** (-gap / 400.0))
    err = actual - pred
    grad = {c: 0.0 for c in confeds}
    count = {c: 0 for c in confeds}
    for c in confeds:
        hmask = hc == c
        amask = ac == c
        grad[c] = err[hmask].sum() - err[amask].sum()
        count[c] = hmask.sum() + amask.sum()
    for c in confeds:
        if count[c]:
            offset[c] += CONFED_FIT_LR * grad[c] / count[c]
    mean_offset = sum(offset.values()) / len(offset)
    offset = {c: offset[c] - mean_offset for c in confeds}

print("\nFitted confederation Elo offsets (centred, + = pool was under-rated):")
for c in sorted(offset, key=offset.get, reverse=True):
    print(f"  {c:<10} {offset[c]:+7.1f}  ({count[c]} match-appearances)")

# ----------------------------------------------------------------------------
# 3. Apply offsets to the World Cup field
# ----------------------------------------------------------------------------
groups_table = pd.read_parquet(GROUPS_PATH)
group_fixtures = pd.read_parquet(GROUP_FIXTURES_PATH)
team_to_group = dict(zip(groups_table["team"], groups_table["group"]))

unmapped = [t for t in team_to_group if t not in team_confederation]
assert not unmapped, f"WC teams with no confederation: {unmapped}"

sim_ratings = {t: ratings[t] + offset[team_confederation[t]] for t in team_to_group}

# Persist the confederation-adjusted WC ratings so the squad-strength blend (10) can build on them.
WC_ADJUSTED_ELO_PATH = PROCESSED_DATA_DIR / "wc_adjusted_elo.parquet"
pd.DataFrame({"team": list(sim_ratings.keys()),
              "raw_elo": [ratings[t] for t in sim_ratings],
              "confederation": [team_confederation[t] for t in sim_ratings],
              "adjusted_elo": list(sim_ratings.values())}).to_parquet(WC_ADJUSTED_ELO_PATH, index=False)
SUPREMACY_PARAMS_PATH = PROCESSED_DATA_DIR / "supremacy_params.parquet"
pd.DataFrame([{"slope": slope, "intercept": intercept, "total_goals": total_goals}]).to_parquet(SUPREMACY_PARAMS_PATH, index=False)

# ----------------------------------------------------------------------------
# 4. Elo -> expected goals -> match probabilities (uses adjusted sim_ratings)
# ----------------------------------------------------------------------------
HOST_NATIONS = {"United States", "Mexico", "Canada"}   # only hosts get home advantage; all other WC games are neutral


def elo_expected_goals(team_a, team_b, neutral=False):
    gap = sim_ratings[team_a] - sim_ratings[team_b]
    home_advantage = 0.0 if (neutral or team_a not in HOST_NATIONS) else intercept
    supremacy = slope * gap + home_advantage
    expected_a = max((total_goals + supremacy) / 2.0, 0.05)
    expected_b = max((total_goals - supremacy) / 2.0, 0.05)
    return expected_a, expected_b


@lru_cache(maxsize=None)
def win_probability(team_a, team_b):
    expected_a, expected_b = elo_expected_goals(team_a, team_b, neutral=True)
    return engine_win_probability(expected_a, expected_b)


def knockout_winner(team_a, team_b, rng):
    return team_a if rng.random() < win_probability(team_a, team_b) else team_b


# ----------------------------------------------------------------------------
# 5. Group stage (vectorised) & bracket (identical to nb 08)
# ----------------------------------------------------------------------------
team_list = list(team_to_group.keys())
group_of_index = [team_to_group[t] for t in team_list]
home_idx = np.array([team_list.index(t) for t in group_fixtures["home_team"]])
away_idx = np.array([team_list.index(t) for t in group_fixtures["away_team"]])
expected_home_goals = np.array([elo_expected_goals(h, a)[0]
                                for h, a in zip(group_fixtures["home_team"], group_fixtures["away_team"])])
expected_away_goals = np.array([elo_expected_goals(h, a)[1]
                                for h, a in zip(group_fixtures["home_team"], group_fixtures["away_team"])])
group_to_indices = {g: [i for i, gi in enumerate(group_of_index) if gi == g]
                    for g in sorted(set(group_of_index))}


group_stage_sampler = build_scoreline_sampler(expected_home_goals, expected_away_goals)


def simulate_qualifiers(rng):
    home_goals, away_goals = sample_scorelines(group_stage_sampler, rng)
    points = np.zeros(len(team_list))
    gf = np.zeros(len(team_list))
    ga = np.zeros(len(team_list))
    np.add.at(gf, home_idx, home_goals)
    np.add.at(gf, away_idx, away_goals)
    np.add.at(ga, home_idx, away_goals)
    np.add.at(ga, away_idx, home_goals)
    hw = home_goals > away_goals
    aw = away_goals > home_goals
    dr = home_goals == away_goals
    np.add.at(points, home_idx[hw], 3)
    np.add.at(points, away_idx[aw], 3)
    np.add.at(points, home_idx[dr], 1)
    np.add.at(points, away_idx[dr], 1)
    gd = gf - ga

    def rank_key(i):
        return (points[i], gd[i], gf[i])

    winners, runners, third_candidates = {}, {}, []
    for group, indices in group_to_indices.items():
        ordered = sorted(indices, key=rank_key, reverse=True)
        winners[group] = ordered[0]
        runners[group] = ordered[1]
        third_candidates.append((group, ordered[2]))
    best_thirds = sorted(third_candidates, key=lambda gi: rank_key(gi[1]), reverse=True)[:8]
    return ({g: team_list[i] for g, i in winners.items()},
            {g: team_list[i] for g, i in runners.items()},
            {g: team_list[i] for g, i in best_thirds})


# ----------------------------------------------------------------------------
# 6. Run the Monte Carlo
# ----------------------------------------------------------------------------
ROUND_NAMES = ["round_of_16", "quarter_final", "semi_final", "final", "champion"]


def simulate_tournament(rng):
    winners, runners, best_thirds = simulate_qualifiers(rng)
    current = build_round_of_32(winners, runners, best_thirds)
    reached = {}
    round_index = 0
    while len(current) >= 1:
        round_winners = [knockout_winner(a, b, rng) for a, b in current]
        for team in round_winners:
            reached[team] = round_index
        if len(round_winners) == 1:
            break
        current = [(round_winners[i], round_winners[i + 1]) for i in range(0, len(round_winners), 2)]
        round_index += 1
    return reached


rng = np.random.default_rng(RANDOM_SEED)
milestones = {name: {team: 0 for team in team_list} for name in ROUND_NAMES}
for _ in range(N_SIMULATIONS):
    reached = simulate_tournament(rng)
    for team, deepest in reached.items():
        for r in range(deepest + 1):
            milestones[ROUND_NAMES[r]][team] += 1

champion_odds = pd.DataFrame({n: pd.Series(c) for n, c in milestones.items()})
champion_odds = (champion_odds / N_SIMULATIONS * 100).round(1)
champion_odds = champion_odds.sort_values("champion", ascending=False)
print(f"\nRan {N_SIMULATIONS} tournaments on the confederation-adjusted Elo engine")

# ----------------------------------------------------------------------------
# 7. Before/after comparison + save
# ----------------------------------------------------------------------------
before = pd.read_parquet(CHAMPION_ODDS_ELO_PATH)
before_champion = before["Champion %"] if "Champion %" in before.columns else before.iloc[:, -1]

compare = pd.DataFrame({
    "adjusted_champion": champion_odds["champion"],
    "elo_champion": before_champion,
}).fillna(0.0)
compare["delta"] = (compare["adjusted_champion"] - compare["elo_champion"]).round(1)
compare = compare.sort_values("adjusted_champion", ascending=False)

display_cols = {c: c.replace("_", " ").title() + " %" for c in champion_odds.columns}
champion_odds_named = champion_odds.rename(columns=display_cols)
champion_odds_named.to_parquet(CHAMPION_ODDS_ADJ_PATH)
pd.Series(offset).to_frame("elo_offset").to_parquet(CONFED_OFFSETS_PATH)

print("\nTop 16 - confederation-adjusted champion %:")
print(champion_odds_named.head(16).to_string())

# Readable markdown report
lines = []
lines.append("# FIFA WC 2026 - Champion Forecast (Confederation-Adjusted Elo, 10,000 sims)")
lines.append("")
lines.append("> Fix #1: per-confederation Elo offsets fitted from inter-confederation matches "
             f"since {CONFED_FIT_FROM_YEAR}, then applied to the World Cup field. Corrects the "
             "non-European over-rating the eye test flagged. Same simulation engine as nb 08.")
lines.append("")
lines.append("## Fitted confederation offsets (Elo points, + = pool was under-rated by raw Elo)")
lines.append("")
lines.append("| Confederation | Elo offset |")
lines.append("|---|---|")
for c in sorted(offset, key=offset.get, reverse=True):
    lines.append(f"| {c} | {offset[c]:+.1f} |")
lines.append("")
lines.append("## Title favourites (top 16, adjusted)")
lines.append("")
lines.append("| Team | Round Of 16 % | Quarter Final % | Semi Final % | Final % | Champion % |")
lines.append("|---|---|---|---|---|---|")
for team, row in champion_odds.head(16).iterrows():
    lines.append(f"| {team} | {row['round_of_16']} | {row['quarter_final']} | "
                 f"{row['semi_final']} | {row['final']} | {row['champion']} |")
lines.append("")
lines.append("## Before vs after the confederation fix (champion %)")
lines.append("")
lines.append("| Team | Elo (nb 08) | Adjusted | Delta |")
lines.append("|---|---|---|---|")
for team, row in compare.head(20).iterrows():
    lines.append(f"| {team} | {row['elo_champion']} | {row['adjusted_champion']} | {row['delta']:+.1f} |")
lines.append("")
REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
print(f"\nWrote {REPORT_PATH.name}, {CHAMPION_ODDS_ADJ_PATH.name}, {CONFED_OFFSETS_PATH.name}")
