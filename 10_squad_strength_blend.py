"""
Notebook 10 (script form) - Squad-strength talent prior.

Why: the confederation offset (script 09) is confederation-level, so it can't tell
that Japan is AFC's outlier - it over-punished Japan along with the rest of Asia.
A squad-strength rating is TEAM-level talent and cuts across that blindness: it lifts
Japan back up on merit, and gives an independent talent check on every team.

Source: EA Sports FC 26 player ratings (full rosters, ~18k players, Sept 2025) - the
freshest squads for the 2026 World Cup. (FC24 = the earlier fallback.)
Squad strength = 0.7 * mean(best 11 by overall) + 0.3 * mean(top 23). Best-XI
weighted so a strong spine matters more than bench depth.

Blend: convert squad strength to an Elo-equivalent by regressing the
confederation-adjusted Elo on squad strength (over well-covered teams), then blend
final = (1-W) * form_elo + W * squad_implied_elo. Teams with thin FC24 data stay
Elo-only (no squad pull). Re-run the same 10k simulation.

Run: .venv/Scripts/python.exe fifa_wc_2026_poisson/10_squad_strength_blend.py
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
# Config
# ----------------------------------------------------------------------------
PROCESSED_DATA_DIR = Path(r"D:/Data Science/Visual Studio Code/fifa_wc_2026_poisson/data/processed")
WC_ADJUSTED_ELO_PATH = PROCESSED_DATA_DIR / "wc_adjusted_elo.parquet"
SUPREMACY_PARAMS_PATH = PROCESSED_DATA_DIR / "supremacy_params.parquet"
GROUPS_PATH = PROCESSED_DATA_DIR / "wc_groups.parquet"
GROUP_FIXTURES_PATH = PROCESSED_DATA_DIR / "wc_group_fixtures.parquet"
CHAMPION_ODDS_ELO_PATH = PROCESSED_DATA_DIR / "champion_odds_elo.parquet"            # raw Elo (nb 08)
CHAMPION_ODDS_ADJ_PATH = PROCESSED_DATA_DIR / "champion_odds_elo_adjusted.parquet"   # +confederation (09)
CHAMPION_ODDS_FINAL_PATH = PROCESSED_DATA_DIR / "champion_odds_final.parquet"
SQUAD_STRENGTH_PATH = PROCESSED_DATA_DIR / "squad_strength_fc26.parquet"
REPORT_PATH = Path(r"D:/Data Science/Visual Studio Code/fifa_wc_2026_poisson/CHAMPION_ODDS_FINAL.md")
# Talent prior source. FC26 (Sept-2025 sofifa snapshot) = the freshest squads for the 2026 WC;
# swap FC_PLAYERS_PATH + FIFA_VERSION to fall back to FC24 (FC_15_24_male_players.csv, version 24).
FC_PLAYERS_PATH = Path(r"D:/Data Science/Datasets/Football/FC_26_male_players.csv")
FIFA_VERSION = 26

MAX_GOALS = 10
RANDOM_SEED = 2026
N_SIMULATIONS = 10000

# Both knobs FITTED by a continuous optimizer (scipy minimize_scalar, bounds [0,1]) on
# held-out RPS in 13_fit_presets.py - NO hand-set candidate menu; the data walks to the
# optimum. Test set = 445 internationals after the FC26 snapshot (2025-09-19). Fresher FC26
# squads carry more signal than FC24 (held-out gain +2.63% vs +0.86%). HONEST READING: the
# RPS surface is flat - any blend in 0.40-0.65 is within noise, so this is really "talent
# weight ~0.5"; the exact decimal is not signal the 445-match sample supports. Values below
# are the optimizer's locked point. See FITTED_PRESETS.md.
SQUAD_BLEND_WEIGHT = 0.53      # talent vs form; optimizer point (~0.5, flat 0.40-0.65). Was hand-set 0.35.
BEST_XI_WEIGHT = 0.63         # best-XI vs top-23 depth; optimizer point (surface ~flat, immaterial)
MIN_PLAYERS_RELIABLE = 15     # below this, FC24 coverage too thin -> Elo only

# sofifa national-team spellings -> our WC team names
FC_NATION_TO_WC = {
    "Korea Republic": "South Korea",
    "Côte d'Ivoire": "Ivory Coast",
    "Congo DR": "DR Congo",
    "Cape Verde Islands": "Cape Verde",   # FC24 spelling
    "Cabo Verde": "Cape Verde",           # FC26 spelling
    "Curacao": "Curaçao",
    "Czechia": "Czech Republic",          # FC26 names it Czechia
    "Türkiye": "Turkey",                  # FC26 names it Türkiye
}

# ----------------------------------------------------------------------------
# 1. Load confederation-adjusted ratings + supremacy params
# ----------------------------------------------------------------------------
wc = pd.read_parquet(WC_ADJUSTED_ELO_PATH)
form_elo = dict(zip(wc["team"], wc["adjusted_elo"]))
params = pd.read_parquet(SUPREMACY_PARAMS_PATH).iloc[0]
slope, intercept, total_goals = params["slope"], params["intercept"], params["total_goals"]

groups_table = pd.read_parquet(GROUPS_PATH)
group_fixtures = pd.read_parquet(GROUP_FIXTURES_PATH)
team_to_group = dict(zip(groups_table["team"], groups_table["group"]))

# ----------------------------------------------------------------------------
# 2. Squad strength from FC26
# ----------------------------------------------------------------------------
fc_players = pd.read_csv(FC_PLAYERS_PATH, low_memory=False)
fc_players = fc_players[fc_players["fifa_version"] == FIFA_VERSION].copy()
fc_players["wc_team"] = fc_players["nationality_name"].replace(FC_NATION_TO_WC)

# Best XI by formation (4-3-3), so squad strength reflects a realistic line-up
# rather than the 11 highest-rated names (which could be 3 strikers + 0 left-backs).
DEFENDERS = {"CB", "LB", "RB", "LWB", "RWB"}
MIDFIELDERS = {"CDM", "CM", "CAM", "LM", "RM"}
FORWARDS = {"LW", "RW", "ST", "CF"}
FORMATION = {"GK": 1, "DEF": 4, "MID": 3, "FWD": 3}


def position_bucket(primary_position):
    if primary_position == "GK":
        return "GK"
    if primary_position in DEFENDERS:
        return "DEF"
    if primary_position in MIDFIELDERS:
        return "MID"
    return "FWD"


fc_players["primary_pos"] = fc_players["player_positions"].str.split(",").str[0].str.strip()
fc_players["bucket"] = fc_players["primary_pos"].map(position_bucket)

squad_rows = []
for team in team_to_group:
    sub = fc_players[fc_players["wc_team"] == team]
    n = len(sub)
    if n == 0:
        strength = np.nan
    else:
        best_xi = [ovr for bucket, slots in FORMATION.items()
                   for ovr in sub.loc[sub["bucket"] == bucket, "overall"].nlargest(slots)]
        best_xi_mean = np.mean(best_xi)            # may be < 11 players for thin squads (Elo-only anyway)
        depth23 = sub["overall"].nlargest(23).mean()
        strength = BEST_XI_WEIGHT * best_xi_mean + (1 - BEST_XI_WEIGHT) * depth23
    squad_rows.append({"team": team, "squad_strength": strength, "n_players": n,
                       "reliable": n >= MIN_PLAYERS_RELIABLE})

squad = pd.DataFrame(squad_rows)
squad.to_parquet(SQUAD_STRENGTH_PATH, index=False)
print(f"Squad strength computed for {squad['squad_strength'].notna().sum()}/48 teams "
      f"({squad['reliable'].sum()} reliable, n>={MIN_PLAYERS_RELIABLE})")
print("Thin coverage (Elo only):", squad.loc[~squad["reliable"], "team"].tolist())

# ----------------------------------------------------------------------------
# 3. Squad strength -> Elo-implied (regress on reliable teams), then blend
# ----------------------------------------------------------------------------
rel = squad[squad["reliable"]].copy()
rel["form_elo"] = rel["team"].map(form_elo)
s_slope, s_intercept = np.polyfit(rel["squad_strength"], rel["form_elo"], 1)
print(f"\nElo-per-OVR fit: form_elo = {s_slope:.1f} * squad + {s_intercept:.0f}")

final_elo = {}
for row in squad.itertuples(index=False):
    fe = form_elo[row.team]
    if row.reliable:
        squad_implied = s_slope * row.squad_strength + s_intercept
        final_elo[row.team] = (1 - SQUAD_BLEND_WEIGHT) * fe + SQUAD_BLEND_WEIGHT * squad_implied
    else:
        final_elo[row.team] = fe

sim_ratings = final_elo

# Persist the final blended ratings so the example-tournament script (11) can replay one readable run.
FINAL_RATINGS_PATH = PROCESSED_DATA_DIR / "wc_final_ratings.parquet"
pd.DataFrame({"team": list(final_elo.keys()),
              "final_elo": list(final_elo.values())}).to_parquet(FINAL_RATINGS_PATH, index=False)

# ----------------------------------------------------------------------------
# 4. Elo -> expected goals -> match probabilities (identical engine)
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
# 5. Group stage (vectorised) & bracket (identical to nb 08 / 09)
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
print(f"\nRan {N_SIMULATIONS} tournaments on the squad-blended engine")

# ----------------------------------------------------------------------------
# 7. Three-way comparison + save
# ----------------------------------------------------------------------------
raw_elo = pd.read_parquet(CHAMPION_ODDS_ELO_PATH)
adj_elo = pd.read_parquet(CHAMPION_ODDS_ADJ_PATH)
raw_c = raw_elo["Champion %"] if "Champion %" in raw_elo.columns else raw_elo.iloc[:, -1]
adj_c = adj_elo["Champion %"] if "Champion %" in adj_elo.columns else adj_elo.iloc[:, -1]

compare = pd.DataFrame({
    "elo": raw_c,
    "plus_confed": adj_c,
    "plus_squad": champion_odds["champion"],
}).fillna(0.0)
compare = compare.sort_values("plus_squad", ascending=False)

display_cols = {c: c.replace("_", " ").title() + " %" for c in champion_odds.columns}
champion_odds_named = champion_odds.rename(columns=display_cols)
champion_odds_named.to_parquet(CHAMPION_ODDS_FINAL_PATH)

print("\nTop 16 - final (Elo + confederation + squad):")
print(champion_odds_named.head(16).to_string())
print("\nThree-stage champion % (top 16):")
print(compare.head(16).round(1).to_string())

squad_lookup = dict(zip(squad["team"], squad["squad_strength"]))
lines = []
lines.append("# FIFA WC 2026 - Champion Forecast (FINAL: Elo + Confederation + Squad strength)")
lines.append("")
lines.append("> Three corrections stacked on the raw Elo Monte Carlo (nb 08): "
             "(1) confederation calibration from inter-confederation results, "
             "(2) FC26 squad-strength talent prior (best-XI weighted), blended "
             f"{int(SQUAD_BLEND_WEIGHT*100)}% talent / {int((1-SQUAD_BLEND_WEIGHT)*100)}% form. "
             "Same 10,000-simulation engine throughout.")
lines.append("")
lines.append("## Title favourites (top 16)")
lines.append("")
lines.append("| Team | Squad (FC26) | Round Of 16 % | Quarter Final % | Semi Final % | Final % | Champion % |")
lines.append("|---|---|---|---|---|---|---|")
for team, row in champion_odds.head(16).iterrows():
    ss = squad_lookup.get(team)
    ss_str = f"{ss:.1f}" if pd.notna(ss) else "n/a"
    lines.append(f"| {team} | {ss_str} | {row['round_of_16']} | {row['quarter_final']} | "
                 f"{row['semi_final']} | {row['final']} | {row['champion']} |")
lines.append("")
lines.append("## Champion % across the three correction stages")
lines.append("")
lines.append("| Team | Elo (nb08) | +Confederation | +Squad (final) |")
lines.append("|---|---|---|---|")
for team, row in compare.head(20).iterrows():
    lines.append(f"| {team} | {row['elo']:.1f} | {row['plus_confed']:.1f} | {row['plus_squad']:.1f} |")
lines.append("")
REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
print(f"\nWrote {REPORT_PATH.name}, {CHAMPION_ODDS_FINAL_PATH.name}, {SQUAD_STRENGTH_PATH.name}")
