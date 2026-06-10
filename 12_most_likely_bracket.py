"""
Notebook 12 (script form) - The "most-likely" predicted bracket (curiosity / comparison).

The 10k Monte Carlo (09/10) gives PROBABILITIES, not a single bracket. This script
collapses those probabilities into ONE deterministic predicted bracket, two ways combined:
  - GROUP STAGE: aggregate. Tally how often each team finishes 1st/2nd/3rd across 10k
    runs, then rank each group by expected finishing position. Predicted qualifiers =
    12 most-likely winners + 12 most-likely runners-up + the 8 teams most often among
    the best third-placed.
  - KNOCKOUTS: chalk. From the predicted Round of 32, advance the higher win-probability
    team in every tie (no dice) down to a predicted champion.

Then it compares this single bracket against the 10k forecast (champion_odds_final) to
show where "most-likely" agrees with the probability table and where it diverges.

This does NOT change the frozen model - it only reads its ratings and re-tallies.
Run: .venv/Scripts/python.exe fifa_wc_2026_poisson/12_most_likely_bracket.py
"""

import pandas as pd
import numpy as np
from functools import lru_cache
from pathlib import Path

from match_engine import (win_probability as engine_win_probability,
                          build_scoreline_sampler, sample_scorelines,
                          build_round_of_32)

PROCESSED_DATA_DIR = Path(r"D:/Data Science/Visual Studio Code/fifa_wc_2026_poisson/data/processed")
FINAL_RATINGS_PATH = PROCESSED_DATA_DIR / "wc_final_ratings.parquet"
SUPREMACY_PARAMS_PATH = PROCESSED_DATA_DIR / "supremacy_params.parquet"
GROUPS_PATH = PROCESSED_DATA_DIR / "wc_groups.parquet"
GROUP_FIXTURES_PATH = PROCESSED_DATA_DIR / "wc_group_fixtures.parquet"
CHAMPION_ODDS_FINAL_PATH = PROCESSED_DATA_DIR / "champion_odds_final.parquet"
REPORT_PATH = Path(r"D:/Data Science/Visual Studio Code/fifa_wc_2026_poisson/MOST_LIKELY_BRACKET.md")

RANDOM_SEED = 2026
N_SIMULATIONS = 10000

# ----------------------------------------------------------------------------
# 1. Load the frozen ratings (same final ratings the forecast was built on)
# ----------------------------------------------------------------------------
ratings = dict(pd.read_parquet(FINAL_RATINGS_PATH).itertuples(index=False, name=None))
params = pd.read_parquet(SUPREMACY_PARAMS_PATH).iloc[0]
slope, intercept, total_goals = params["slope"], params["intercept"], params["total_goals"]
groups_table = pd.read_parquet(GROUPS_PATH)
group_fixtures = pd.read_parquet(GROUP_FIXTURES_PATH)
team_to_group = dict(zip(groups_table["team"], groups_table["group"]))


HOST_NATIONS = {"United States", "Mexico", "Canada"}   # only hosts get home advantage; all other WC games are neutral


def expected_goals(team_a, team_b, neutral=False):
    gap = ratings[team_a] - ratings[team_b]
    home_advantage = 0.0 if (neutral or team_a not in HOST_NATIONS) else intercept
    supremacy = slope * gap + home_advantage
    return max((total_goals + supremacy) / 2.0, 0.05), max((total_goals - supremacy) / 2.0, 0.05)


@lru_cache(maxsize=None)
def win_probability(team_a, team_b):
    return engine_win_probability(*expected_goals(team_a, team_b, neutral=True))


# ----------------------------------------------------------------------------
# 2. Group-stage setup (same vectorised engine as scripts 09/10)
# ----------------------------------------------------------------------------
team_list = list(team_to_group.keys())
group_of_index = [team_to_group[t] for t in team_list]
home_idx = np.array([team_list.index(t) for t in group_fixtures["home_team"]])
away_idx = np.array([team_list.index(t) for t in group_fixtures["away_team"]])
expected_home_goals = np.array([expected_goals(h, a)[0]
                                for h, a in zip(group_fixtures["home_team"], group_fixtures["away_team"])])
expected_away_goals = np.array([expected_goals(h, a)[1]
                                for h, a in zip(group_fixtures["home_team"], group_fixtures["away_team"])])
group_to_indices = {g: [i for i, gi in enumerate(group_of_index) if gi == g]
                    for g in sorted(set(group_of_index))}
group_stage_sampler = build_scoreline_sampler(expected_home_goals, expected_away_goals)

# ----------------------------------------------------------------------------
# 3. Run 10k group stages, tally finishing positions + best-third qualification
# ----------------------------------------------------------------------------
n_teams = len(team_list)
position_tally = np.zeros((n_teams, 4))      # [team, finish 1st/2nd/3rd/4th]
group_third_qualify = {g: 0 for g in group_to_indices}   # times this group's 3rd makes the best-8
rng = np.random.default_rng(RANDOM_SEED)

for _ in range(N_SIMULATIONS):
    home_goals, away_goals = sample_scorelines(group_stage_sampler, rng)
    points = np.zeros(n_teams); gf = np.zeros(n_teams); ga = np.zeros(n_teams)
    np.add.at(gf, home_idx, home_goals); np.add.at(gf, away_idx, away_goals)
    np.add.at(ga, home_idx, away_goals); np.add.at(ga, away_idx, home_goals)
    hw = home_goals > away_goals; aw = away_goals > home_goals; dr = home_goals == away_goals
    np.add.at(points, home_idx[hw], 3); np.add.at(points, away_idx[aw], 3)
    np.add.at(points, home_idx[dr], 1); np.add.at(points, away_idx[dr], 1)
    gd = gf - ga

    def rank_key(i):
        return (points[i], gd[i], gf[i])

    third_candidates = []
    for group, indices in group_to_indices.items():
        ordered = sorted(indices, key=rank_key, reverse=True)
        for finishing_position, team_index in enumerate(ordered):
            position_tally[team_index, finishing_position] += 1
        third_candidates.append((group, ordered[2]))
    best_thirds = sorted(third_candidates, key=lambda gi: rank_key(gi[1]), reverse=True)[:8]
    for group, _ in best_thirds:
        group_third_qualify[group] += 1

position_prob = position_tally / N_SIMULATIONS          # P(finish kth)
expected_position = position_prob @ np.array([1, 2, 3, 4])   # lower = better

# ----------------------------------------------------------------------------
# 4. Build the most-likely qualifiers + predicted standings
# ----------------------------------------------------------------------------
predicted_standings = {}     # group -> list of team names, best->worst by expected position
winners, runners, predicted_third = {}, {}, {}
for group, indices in group_to_indices.items():
    ordered = sorted(indices, key=lambda i: expected_position[i])
    predicted_standings[group] = [team_list[i] for i in ordered]
    winners[group] = team_list[ordered[0]]
    runners[group] = team_list[ordered[1]]
    predicted_third[group] = team_list[ordered[2]]

# 8 best thirds = the 8 GROUPS whose 3rd-placed team most often makes the best-8 cut over
# 10k (each run ranks the twelve 3rd-placed teams by points/GD/GF = FIFA's exact rule).
qualifying_third_groups = sorted(group_third_qualify, key=group_third_qualify.get, reverse=True)[:8]
thirds_by_group = {g: predicted_third[g] for g in qualifying_third_groups}

predicted_round_of_32 = build_round_of_32(winners, runners, thirds_by_group)

# ----------------------------------------------------------------------------
# 5. Chalk knockout - advance the higher win-probability team every tie
# ----------------------------------------------------------------------------
ROUND_NAMES = ["Round of 16", "Quarter-finals", "Semi-finals", "Final"]
chalk_rounds = []
current = predicted_round_of_32
chalk_rounds.append(("Round of 32", list(current)))
round_pointer = 0
while len(current) > 1:
    advancing = [a if win_probability(a, b) >= 0.5 else b for a, b in current]
    if len(advancing) == 1:
        break
    current = [(advancing[i], advancing[i + 1]) for i in range(0, len(advancing), 2)]
    chalk_rounds.append((ROUND_NAMES[round_pointer], list(current)))
    round_pointer += 1
final_a, final_b = chalk_rounds[-1][1][0]
predicted_champion = final_a if win_probability(final_a, final_b) >= 0.5 else final_b

# ----------------------------------------------------------------------------
# 6. Compare against the 10k forecast
# ----------------------------------------------------------------------------
forecast = pd.read_parquet(CHAMPION_ODDS_FINAL_PATH)
champion_col = [c for c in forecast.columns if "Champion" in c][0]
forecast_top = forecast[champion_col].sort_values(ascending=False)
chalk_final_four = [a for a, _ in chalk_rounds[-2][1]] + [b for _, b in chalk_rounds[-2][1]]

# ----------------------------------------------------------------------------
# 7. Write the readable report
# ----------------------------------------------------------------------------
lines = []
lines.append("# FIFA WC 2026 - Most-Likely Predicted Bracket (vs the 10k forecast)")
lines.append("")
lines.append("> ONE deterministic bracket, for comparison only. Groups = aggregate "
             "(ranked by expected finishing position over 10,000 runs); knockouts = chalk "
             "(higher win-probability team advances every tie). This is a *simplification* "
             "of the probability table, not a replacement for it.")
lines.append("")
lines.append(f"## Predicted champion (chalk path): **{predicted_champion}**")
lines.append("")

lines.append("## Predicted group standings (by expected finish over 10k)")
for group in sorted(predicted_standings):
    lines.append("")
    lines.append(f"### Group {group}")
    lines.append("")
    lines.append("| # | Team | P(1st) | P(2nd) | P(3rd) | Status |")
    lines.append("|---|---|---|---|---|---|")
    for rank, team in enumerate(predicted_standings[group], 1):
        i = team_list.index(team)
        if rank == 1:
            status = "Q (winner)"
        elif rank == 2:
            status = "Q (runner-up)"
        elif rank == 3:
            status = "Q (best 3rd)" if group in qualifying_third_groups else "3rd - out"
        else:
            status = "out"
        lines.append(f"| {rank} | {team} | {position_prob[i,0]:.0%} | {position_prob[i,1]:.0%} "
                     f"| {position_prob[i,2]:.0%} | {status} |")

lines.append("")
lines.append("## Predicted Round of 32")
lines.append("")
for a, b in predicted_round_of_32:
    lines.append(f"- {a} vs {b}")

lines.append("")
lines.append("## Chalk knockout path")
for round_name, ties in chalk_rounds[1:]:
    lines.append("")
    lines.append(f"### {round_name}")
    lines.append("")
    for a, b in ties:
        favourite = a if win_probability(a, b) >= 0.5 else b
        lines.append(f"- {a} vs {b} -> {favourite} ({max(win_probability(a,b), 1-win_probability(a,b)):.0%})")
lines.append("")
lines.append(f"**Predicted champion: {predicted_champion}**")

lines.append("")
lines.append("## Comparison vs the 10k forecast")
lines.append("")
lines.append(f"- **Predicted champion (chalk):** {predicted_champion} "
             f"| **10k forecast #1:** {forecast_top.index[0]} ({forecast_top.iloc[0]:.1f}%)")
lines.append(f"- **Predicted final four (chalk):** {', '.join(chalk_final_four)}")
lines.append("- **10k forecast top 8 by champion %:** "
             + ", ".join(f"{t} {v:.1f}%" for t, v in forecast_top.head(8).items()))
REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

print(f"Predicted champion (chalk): {predicted_champion}")
print(f"10k forecast #1: {forecast_top.index[0]} ({forecast_top.iloc[0]:.1f}%)")
print(f"Chalk final four: {chalk_final_four}")
print(f"Wrote {REPORT_PATH.name}")
