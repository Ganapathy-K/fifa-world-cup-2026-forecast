"""
Notebook 11 (script form) - One example tournament, written out in full.

The 10k Monte Carlo only reports aggregate odds. This replays a SINGLE tournament
on the final ratings and writes every concrete detail: each group's table
(P/W/D/L/GF/GA/GD/Pts) with the actual match scorelines, then the knockout bracket
filled in round by round to the champion.

IMPORTANT: this is ONE of the 10,000 simulated tournaments - it is illustrative,
NOT the forecast. A single run is luck (like one game of snakes & ladders); the
forecast is the aggregate over 10,000 runs. Re-running with a different seed gives a
different winner. This file exists to make the machinery concrete and followable.

Run: .venv/Scripts/python.exe fifa_wc_2026_poisson/11_example_tournament.py
"""

import pandas as pd
import numpy as np
from functools import lru_cache
from pathlib import Path

from match_engine import (win_probability as engine_win_probability,
                          sample_scoreline, build_round_of_32)

PROCESSED_DATA_DIR = Path(r"D:/Data Science/Visual Studio Code/fifa_wc_2026_poisson/data/processed")
FINAL_RATINGS_PATH = PROCESSED_DATA_DIR / "wc_final_ratings.parquet"
SUPREMACY_PARAMS_PATH = PROCESSED_DATA_DIR / "supremacy_params.parquet"
GROUPS_PATH = PROCESSED_DATA_DIR / "wc_groups.parquet"
GROUP_FIXTURES_PATH = PROCESSED_DATA_DIR / "wc_group_fixtures.parquet"
REPORT_PATH = Path(r"D:/Data Science/Visual Studio Code/fifa_wc_2026_poisson/EXAMPLE_TOURNAMENT_FINAL.md")

MAX_GOALS = 10
EXAMPLE_SEED = 2026

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
def penalty_win_prob(team_a, team_b):
    expected_a, expected_b = expected_goals(team_a, team_b, neutral=True)
    return engine_win_probability(expected_a, expected_b)


def play_match(team_a, team_b, rng, neutral=False):
    expected_a, expected_b = expected_goals(team_a, team_b, neutral=neutral)
    return sample_scoreline(expected_a, expected_b, rng)


def knockout(team_a, team_b, rng):
    """Return (winner, scoreline_string). A draw goes to a penalty shootout."""
    goals_a, goals_b = play_match(team_a, team_b, rng, neutral=True)
    if goals_a > goals_b:
        return team_a, f"{team_a} {goals_a}-{goals_b} {team_b}"
    if goals_b > goals_a:
        return team_b, f"{team_a} {goals_a}-{goals_b} {team_b}"
    winner = team_a if rng.random() < penalty_win_prob(team_a, team_b) else team_b
    return winner, f"{team_a} {goals_a}-{goals_b} {team_b} ({winner} on pens)"


rng = np.random.default_rng(EXAMPLE_SEED)

# ----------------------------------------------------------------------------
# Group stage: play all 72 fixtures, accumulate tables
# ----------------------------------------------------------------------------
groups = sorted(set(team_to_group.values()))
stats = {t: {"P": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0} for t in team_to_group}
group_matches = {g: [] for g in groups}

for fixture in group_fixtures.itertuples(index=False):
    home, away = fixture.home_team, fixture.away_team
    goals_home, goals_away = play_match(home, away, rng)
    group_matches[team_to_group[home]].append(f"{home} {goals_home}-{goals_away} {away}")
    for team, scored, conceded in [(home, goals_home, goals_away), (away, goals_away, goals_home)]:
        stats[team]["P"] += 1
        stats[team]["GF"] += scored
        stats[team]["GA"] += conceded
        stats[team]["W"] += scored > conceded
        stats[team]["D"] += scored == conceded
        stats[team]["L"] += scored < conceded


def rank_key(team):
    s = stats[team]
    return (s["W"] * 3 + s["D"], s["GF"] - s["GA"], s["GF"])


winners, runners, third_candidates = {}, {}, []
group_order = {}
for g in groups:
    ordered = sorted([t for t in team_to_group if team_to_group[t] == g], key=rank_key, reverse=True)
    group_order[g] = ordered
    winners[g], runners[g] = ordered[0], ordered[1]
    third_candidates.append((g, ordered[2]))
best_thirds = sorted(third_candidates, key=lambda gt: rank_key(gt[1]), reverse=True)[:8]
thirds_by_group = {g: t for g, t in best_thirds}
qualified_thirds = set(thirds_by_group.values())

# ----------------------------------------------------------------------------
# Knockouts: Round of 32 -> Final (same rng stream, one coherent tournament)
# ----------------------------------------------------------------------------
pairs = build_round_of_32(winners, runners, thirds_by_group)

captured_rounds = []
for stage in ["Round of 32", "Round of 16", "Quarter-finals", "Semi-finals", "Final"]:
    results = [knockout(a, b, rng) for a, b in pairs]
    captured_rounds.append((stage, [line for _, line in results]))
    advancing = [winner for winner, _ in results]
    if len(advancing) == 1:
        champion = advancing[0]
        break
    pairs = [(advancing[i], advancing[i + 1]) for i in range(0, len(advancing), 2)]

# ----------------------------------------------------------------------------
# Write the readable report
# ----------------------------------------------------------------------------
lines = []
lines.append("# One Simulated World Cup 2026 (illustrative single run)")
lines.append("")
lines.append(f"> **This is ONE of the 10,000 simulated tournaments, seed {EXAMPLE_SEED}.** It is "
             "illustrative, NOT the forecast. A single run is luck; the forecast is the aggregate "
             "over 10,000 runs (see CHAMPION_ODDS_FINAL.md). Re-run with another seed and a "
             "different team lifts the cup. This file exists to make the pipeline concrete.")
lines.append("")
lines.append(f"## Champion in this run: **{champion}**")
lines.append("")
lines.append("## Group stage")
for g in groups:
    lines.append("")
    lines.append(f"### Group {g}")
    lines.append("")
    lines.append("| # | Team | P | W | D | L | GF | GA | GD | Pts | |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for i, t in enumerate(group_order[g], 1):
        s = stats[t]
        points = s["W"] * 3 + s["D"]
        goal_diff = s["GF"] - s["GA"]
        mark = "Q" if i <= 2 else ("Q (best 3rd)" if t in qualified_thirds else "out")
        lines.append(f"| {i} | {t} | {s['P']} | {s['W']} | {s['D']} | {s['L']} | "
                     f"{s['GF']} | {s['GA']} | {goal_diff:+d} | {points} | {mark} |")
    lines.append("")
    lines.append("Results: " + "; ".join(group_matches[g]))
lines.append("")
lines.append("## Knockout bracket")
for stage, scorelines in captured_rounds:
    lines.append("")
    lines.append(f"### {stage}")
    lines.append("")
    for line in scorelines:
        lines.append(f"- {line}")
lines.append("")
lines.append(f"**Champion: {champion}**")
lines.append("")
REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
print(f"Champion in seed {EXAMPLE_SEED}: {champion}")
print(f"Wrote {REPORT_PATH.name}")
