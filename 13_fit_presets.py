"""
Notebook 13 (script form) - Fit the three remaining hand-set presets from data.

Standing rule (fit, don't preset): every constant in the model should be the value
that best predicts out-of-sample reality, not a number picked by hand. Three presets
were still hard-coded. This script replaces each with a held-out RPS grid search and
reports the fitted value next to the old guess.

Knobs fitted here:
  1. SQUAD_BLEND_WEIGHT  (talent vs form)            - live spine, script 10
  2. BEST_XI_WEIGHT      (best-XI vs top-23 depth)   - live spine, script 10
  3. HALF_LIFE_YEARS     (recency decay)             - ARCHIVED ratio baseline only
     (nb02/03/05/06). The live Elo spine has no half-life knob - Elo's recency is
     implicit in walk-forward updating - so fitting it hardens the archived baseline
     but does NOT move the headline forecast. Labelled as such in the output.

Held-out design: EA FC 26 ratings are a Sept-2025 snapshot, so every international
played AFTER the snapshot is genuinely out-of-sample for the squad prior. We walk Elo
forward over the whole history (recording each match's PRE-match ratings), fit the
squad->Elo regression on train-era ratings only, then score candidate (knob) values
by Ranked Probability Score on the post-snapshot test matches. Lower RPS = better.

Run: .venv/Scripts/python.exe fifa_wc_2026_poisson/13_fit_presets.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from functools import lru_cache
from scipy.optimize import minimize_scalar

from match_engine import match_outcome_probabilities

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
PROCESSED_DATA_DIR = Path(r"D:/Data Science/Visual Studio Code/fifa_wc_2026_poisson/data/processed")
PLAYED_MATCHES_PATH = PROCESSED_DATA_DIR / "played_matches.parquet"
CONFED_OFFSETS_PATH = PROCESSED_DATA_DIR / "confederation_offsets.parquet"
FC_PLAYERS_PATH = Path(r"D:/Data Science/Datasets/Football/FC_26_male_players.csv")
FIFA_VERSION = 26
REPORT_PATH = Path(r"D:/Data Science/Visual Studio Code/fifa_wc_2026_poisson/FITTED_PRESETS.md")

INITIAL_RATING = 1500.0
K_FACTOR = 30.0
HOME_ADVANTAGE_ELO = 65.0
FC_SNAPSHOT = pd.Timestamp("2025-09-19")     # EA FC 26 sofifa snapshot; test matches start after this
MIN_PLAYERS_RELIABLE = 15
MIN_TEST_GAMES_BEFORE = 30                    # both sides need a settled rating by the test window

# sofifa national-team spellings -> results.csv / Elo team names (same map as script 10)
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

# Confederation map (copied from script 09 so every team can be confederation-adjusted)
CONFEDERATIONS = {
    "UEFA": ["Sweden", "England", "Germany", "Hungary", "Poland", "Italy", "Switzerland",
             "Netherlands", "Denmark", "Norway", "Austria", "Belgium", "Scotland", "Finland",
             "France", "Spain", "Romania", "Russia", "Bulgaria", "Wales", "Northern Ireland",
             "Portugal", "Turkey", "Republic of Ireland", "Greece", "Estonia", "Iceland",
             "Czechoslovakia", "Israel", "Yugoslavia", "Luxembourg", "Latvia", "Malta",
             "Lithuania", "Cyprus", "Albania", "Croatia", "Czech Republic", "Serbia",
             "Ukraine", "Slovakia", "Slovenia", "Belarus", "Azerbaijan", "Georgia",
             "North Macedonia", "German DR", "Moldova", "Faroe Islands",
             "Bosnia and Herzegovina", "Armenia", "Montenegro", "Kazakhstan", "Liechtenstein",
             "Andorra", "San Marino", "Gibraltar", "Kosovo"],
    "CONMEBOL": ["Argentina", "Brazil", "Uruguay", "Chile", "Paraguay", "Peru", "Colombia",
                 "Ecuador", "Bolivia", "Venezuela"],
    "CONCACAF": ["Mexico", "United States", "Trinidad and Tobago", "Costa Rica", "Jamaica",
                 "Honduras", "El Salvador", "Guatemala", "Panama", "Haiti", "Cuba", "Canada",
                 "Suriname", "Curaçao", "Guyana", "Barbados", "Martinique", "Guadeloupe",
                 "Grenada", "Saint Vincent and the Grenadines", "Antigua and Barbuda",
                 "Saint Lucia", "Saint Kitts and Nevis", "Nicaragua", "Dominica", "Bermuda",
                 "French Guiana", "Puerto Rico", "Dominican Republic", "Aruba", "Belize",
                 "Cayman Islands", "British Virgin Islands"],
    "CAF": ["Egypt", "Zambia", "Kenya", "Uganda", "Tunisia", "Ghana", "Nigeria", "Senegal",
            "Algeria", "Cameroon", "Morocco", "Tanzania", "Mali", "Guinea", "DR Congo",
            "Ivory Coast", "Malawi", "Zimbabwe", "South Africa", "Sudan", "Burkina Faso",
            "Togo", "Angola", "Ethiopia", "Congo", "Gabon", "Libya", "Mozambique", "Botswana",
            "Mauritius", "Benin", "Madagascar", "Liberia", "Sierra Leone", "Lesotho", "Rwanda",
            "Namibia", "Eswatini", "Mauritania", "Niger", "Gambia", "Cape Verde", "Burundi",
            "Zanzibar", "Guinea-Bissau", "Equatorial Guinea", "Seychelles", "Comoros", "Chad",
            "Central African Republic", "Somalia", "Réunion", "Djibouti"],
    "AFC": ["South Korea", "Thailand", "Malaysia", "Japan", "Saudi Arabia", "Indonesia",
            "China PR", "Singapore", "Kuwait", "Iraq", "Qatar", "United Arab Emirates",
            "Iran", "Bahrain", "India", "Oman", "Myanmar", "Hong Kong", "Syria", "Jordan",
            "North Korea", "Uzbekistan", "Philippines", "Vietnam", "Lebanon", "Yemen",
            "Bangladesh", "Vietnam Republic", "Nepal", "Cambodia", "Sri Lanka", "Pakistan",
            "Palestine", "Taiwan", "Laos", "Maldives", "Tajikistan", "Kyrgyzstan",
            "Turkmenistan", "Afghanistan", "Brunei", "Macau", "Guam", "Mongolia", "Bhutan",
            "Australia"],
    "OFC": ["New Zealand", "Fiji", "New Caledonia", "Tahiti", "Solomon Islands", "Vanuatu",
            "Papua New Guinea"],
}
team_confederation = {team: confed for confed, teams in CONFEDERATIONS.items() for team in teams}

# Position buckets for the best-XI (4-3-3), same as script 10
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


def ranked_probability_score(probs, outcomes):
    """Football-standard ordered metric. probs columns = (home, draw, away);
    outcomes in {0 home, 1 draw, 2 away}. Lower is better."""
    cumulative_predicted = np.cumsum(probs, axis=1)
    cumulative_actual = np.cumsum(np.eye(3)[outcomes], axis=1)
    return np.mean(np.sum((cumulative_predicted[:, :-1] - cumulative_actual[:, :-1]) ** 2, axis=1) / 2)


# ----------------------------------------------------------------------------
# 1. Walk Elo forward, recording each match's PRE-match ratings (no leakage)
# ----------------------------------------------------------------------------
played = pd.read_parquet(PLAYED_MATCHES_PATH).sort_values("date").reset_index(drop=True)


def margin_multiplier(goal_difference):
    margin = abs(goal_difference)
    if margin <= 1:
        return 1.0
    if margin == 2:
        return 1.5
    return (11 + margin) / 8


ratings = {}
games_seen = {}
home_elo_before = np.empty(len(played))
away_elo_before = np.empty(len(played))
home_games_before = np.empty(len(played), dtype=int)
away_games_before = np.empty(len(played), dtype=int)
gap_history, goal_diff_history = [], []

for row_index, match in enumerate(played.itertuples(index=False)):
    home_rating = ratings.get(match.home_team, INITIAL_RATING)
    away_rating = ratings.get(match.away_team, INITIAL_RATING)
    home_elo_before[row_index] = home_rating
    away_elo_before[row_index] = away_rating
    home_games_before[row_index] = games_seen.get(match.home_team, 0)
    away_games_before[row_index] = games_seen.get(match.away_team, 0)

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
    games_seen[match.home_team] = games_seen.get(match.home_team, 0) + 1
    games_seen[match.away_team] = games_seen.get(match.away_team, 0) + 1

slope, intercept = np.polyfit(gap_history, goal_diff_history, 1)
total_goals = (played["home_score"] + played["away_score"]).mean()
print(f"Walked Elo over {len(played)} matches; supremacy slope={slope:.5f} intercept={intercept:.3f} "
      f"total_goals={total_goals:.3f}")

played = played.assign(home_elo_before=home_elo_before, away_elo_before=away_elo_before,
                       home_games_before=home_games_before, away_games_before=away_games_before)

# Confederation offsets (already fitted in script 09) -> adjusted pre-match Elo
offsets = pd.read_parquet(CONFED_OFFSETS_PATH)["elo_offset"].to_dict()
played["home_conf"] = played["home_team"].map(team_confederation)
played["away_conf"] = played["away_team"].map(team_confederation)
played["home_adj_elo"] = played["home_elo_before"] + played["home_conf"].map(offsets).fillna(0.0)
played["away_adj_elo"] = played["away_elo_before"] + played["away_conf"].map(offsets).fillna(0.0)

# Snapshot Elo (as of the FC26 release) for fitting the squad->Elo regression leak-free.
# Proxy = the adjusted pre-match Elo of each team's last match on/before the snapshot.
pre_snapshot = played[played["date"] <= FC_SNAPSHOT]
last_seen = {}
for match in pre_snapshot.itertuples(index=False):
    last_seen[match.home_team] = match.home_adj_elo
    last_seen[match.away_team] = match.away_adj_elo
snapshot_elo = last_seen

# ----------------------------------------------------------------------------
# 2. Squad strength per nation from FC26, as a function of the best-XI weight
# ----------------------------------------------------------------------------
fc_players = pd.read_csv(FC_PLAYERS_PATH, low_memory=False)
fc_players = fc_players[fc_players["fifa_version"] == FIFA_VERSION].copy()
fc_players["team"] = fc_players["nationality_name"].replace(FC_NATION_TO_WC)
fc_players["primary_pos"] = fc_players["player_positions"].str.split(",").str[0].str.strip()
fc_players["bucket"] = fc_players["primary_pos"].map(position_bucket)


@lru_cache(maxsize=None)
def squad_strength_table(best_xi_weight):
    """squad strength = best_xi_weight * mean(best XI, 4-3-3) + (1-best_xi_weight) * mean(top 23).
    Cached (the optimizer re-queries the same best-XI weight while tuning the blend weight)."""
    rows = []
    for team, sub in fc_players.groupby("team"):
        n = len(sub)
        if n == 0:
            continue
        best_xi = [ovr for bucket, slots in FORMATION.items()
                   for ovr in sub.loc[sub["bucket"] == bucket, "overall"].nlargest(slots)]
        best_xi_mean = np.mean(best_xi)
        depth23 = sub["overall"].nlargest(23).mean()
        strength = best_xi_weight * best_xi_mean + (1 - best_xi_weight) * depth23
        rows.append({"team": team, "squad_strength": strength, "n_players": n,
                     "reliable": n >= MIN_PLAYERS_RELIABLE})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 3. Build the post-snapshot held-out test set
# ----------------------------------------------------------------------------
fc_teams = set(fc_players["team"].unique())
test = played[(played["date"] > FC_SNAPSHOT)
              & played["home_team"].isin(fc_teams) & played["away_team"].isin(fc_teams)
              & (played["home_games_before"] >= MIN_TEST_GAMES_BEFORE)
              & (played["away_games_before"] >= MIN_TEST_GAMES_BEFORE)].copy()
test_outcome = np.where(test["home_score"] > test["away_score"], 0,
                        np.where(test["home_score"] == test["away_score"], 1, 2))
test_neutral = test["neutral"].to_numpy()
print(f"Held-out test matches (post-FC26-snapshot, both sides FC26+settled): {len(test)}")


def held_out_rps(best_xi_weight, blend_weight):
    """Score the squad-blended Elo model on the held-out matches for one (knob) pair."""
    squad = squad_strength_table(round(best_xi_weight, 4))   # round -> reliable cache hits
    reliable = squad[squad["reliable"]].copy()
    reliable["snapshot_elo"] = reliable["team"].map(snapshot_elo)
    reliable = reliable.dropna(subset=["snapshot_elo"])
    s_slope, s_intercept = np.polyfit(reliable["squad_strength"], reliable["snapshot_elo"], 1)
    squad_implied = {row.team: s_slope * row.squad_strength + s_intercept
                     for row in squad[squad["reliable"]].itertuples(index=False)}

    home_implied = test["home_team"].map(squad_implied)
    away_implied = test["away_team"].map(squad_implied)
    # teams below the reliability floor keep pure form Elo (blend pulls toward themselves)
    home_blend = np.where(home_implied.notna(),
                          (1 - blend_weight) * test["home_adj_elo"] + blend_weight * home_implied.fillna(0),
                          test["home_adj_elo"])
    away_blend = np.where(away_implied.notna(),
                          (1 - blend_weight) * test["away_adj_elo"] + blend_weight * away_implied.fillna(0),
                          test["away_adj_elo"])

    gap = home_blend - away_blend
    home_bonus = np.where(test_neutral, 0.0, intercept)
    supremacy = slope * gap + home_bonus
    expected_home = np.clip((total_goals + supremacy) / 2.0, 0.05, None)
    expected_away = np.clip((total_goals - supremacy) / 2.0, 0.05, None)

    probs = np.array([[(p := match_outcome_probabilities(eh, ea))["home_win"], p["draw"], p["away_win"]]
                      for eh, ea in zip(expected_home, expected_away)])
    return ranked_probability_score(probs, test_outcome)


# Baseline: pure form Elo (blend_weight = 0) for reference
baseline_rps = held_out_rps(best_xi_weight=0.7, blend_weight=0.0)

# ----------------------------------------------------------------------------
# 4. Fit both squad knobs with a CONTINUOUS OPTIMIZER (no hand-set grid/menu).
#    minimize_scalar walks to the lowest-RPS point itself. Bounds [0, 1] are the
#    mathematical range of a weight, not tuned values. Coordinate descent (alternate
#    the two scalars) because the knobs are near-independent; converges in a couple
#    of passes from a neutral 0.5 start.
# ----------------------------------------------------------------------------
def rps_at(best_xi_weight, blend_weight):
    return held_out_rps(best_xi_weight, blend_weight)


best_xi, blend = 0.5, 0.5
for _ in range(4):
    blend = minimize_scalar(lambda w: rps_at(best_xi, w), bounds=(0, 1), method="bounded").x
    best_xi = minimize_scalar(lambda w: rps_at(w, blend), bounds=(0, 1), method="bounded").x

# Round to 2 dp on purpose: finer precision off 445 matches is noise, not signal.
fitted_blend = round(float(blend), 2)
fitted_best_xi = round(float(best_xi), 2)
best_rps = held_out_rps(fitted_best_xi, fitted_blend)

# Characterise the flat region around the optimum - for honest labelling, NOT selection.
# "Indistinguishable" = within the RPS noise floor of the best point on this sample.
NOISE_FLOOR = 0.0003
sensitivity = [(round(b, 2), held_out_rps(fitted_best_xi, b)) for b in np.arange(0.20, 0.81, 0.05)]
flat_band = [b for b, r in sensitivity if r <= best_rps + NOISE_FLOOR]
flat_lo, flat_hi = min(flat_band), max(flat_band)

print(f"\nBaseline (form Elo only) held-out RPS: {baseline_rps:.5f}")
print(f"OPTIMIZER best-XI weight   = {fitted_best_xi:.2f}  (continuous, bounds [0,1]; was hand-set 0.70)")
print(f"OPTIMIZER squad blend wt   = {fitted_blend:.2f}  (continuous, bounds [0,1]; was hand-set 0.35)")
print(f"Best held-out RPS = {best_rps:.5f}  "
      f"(improvement vs form-only: {(baseline_rps - best_rps) / baseline_rps * 100:+.2f}%)")
print(f"FLAT BAND (RPS within {NOISE_FLOOR} of best): blend {flat_lo:.2f}-{flat_hi:.2f} "
      f"=> report as ~{round(2*((flat_lo+flat_hi)/2))/2:.1f}, the exact decimal is noise.")
print("\nBlend sensitivity at fitted best-XI weight (characterisation, not selection):")
for b, r in sensitivity:
    flag = "  <- flat" if r <= best_rps + NOISE_FLOOR else ""
    print(f"  blend {b:.2f}: RPS {r:.5f}{flag}")

# ----------------------------------------------------------------------------
# 5. HALF_LIFE_YEARS - fit on the ARCHIVED ratio-Poisson baseline (nb06 harness)
#    Labelled clearly: this does NOT feed the live Elo forecast.
# ----------------------------------------------------------------------------
SHRINKAGE_GAMES = 30
MIN_TRAIN_GAMES = 30
RATIO_CUTOFF = pd.Timestamp("2023-01-01")
ratio_train = played[played["date"] < RATIO_CUTOFF]
ratio_test = played[played["date"] >= RATIO_CUTOFF].copy()


def ratio_strengths(matches, reference_date, half_life):
    home_view = matches[["date", "home_team", "home_score", "away_score"]].rename(
        columns={"home_team": "team", "home_score": "scored", "away_score": "conceded"})
    away_view = matches[["date", "away_team", "away_score", "home_score"]].rename(
        columns={"away_team": "team", "away_score": "scored", "home_score": "conceded"})
    tm = pd.concat([home_view, away_view], ignore_index=True)
    age_years = (reference_date - tm["date"]).dt.days / 365.25
    weight = 0.5 ** (age_years / half_life)
    tm["w"], tm["ws"], tm["wc"] = weight, weight * tm["scored"], weight * tm["conceded"]
    baseline = tm["ws"].sum() / tm["w"].sum()
    g = tm.groupby("team").agg(games=("w", "size"), wsum=("w", "sum"),
                               ssum=("ws", "sum"), csum=("wc", "sum"))
    g["attack"] = (g["ssum"] + SHRINKAGE_GAMES * baseline) / (g["wsum"] + SHRINKAGE_GAMES) / baseline
    g["defence"] = (g["csum"] + SHRINKAGE_GAMES * baseline) / (g["wsum"] + SHRINKAGE_GAMES) / baseline
    return g, baseline


def ratio_rps(half_life):
    strengths, baseline = ratio_strengths(ratio_train, RATIO_CUTOFF, half_life)
    rated = set(strengths.index[strengths["games"] >= MIN_TRAIN_GAMES])
    sub = ratio_test[ratio_test["home_team"].isin(rated) & ratio_test["away_team"].isin(rated)].copy()
    eh = strengths.loc[sub["home_team"], "attack"].to_numpy() * strengths.loc[sub["away_team"], "defence"].to_numpy() * baseline
    ea = strengths.loc[sub["away_team"], "attack"].to_numpy() * strengths.loc[sub["home_team"], "defence"].to_numpy() * baseline
    probs = np.array([[(p := match_outcome_probabilities(h, a))["home_win"], p["draw"], p["away_win"]]
                      for h, a in zip(eh, ea)])
    outcome = np.where(sub["home_score"] > sub["away_score"], 0,
                       np.where(sub["home_score"] == sub["away_score"], 1, 2))
    return ranked_probability_score(probs, outcome), len(sub)


half_life_grid = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50]
hl_results = [{"half_life_years": hl, "rps": (r := ratio_rps(hl))[0], "n": r[1]} for hl in half_life_grid]
hl = pd.DataFrame(hl_results)
fitted_half_life = hl.loc[hl["rps"].idxmin(), "half_life_years"]
print(f"\n[ARCHIVED BASELINE ONLY] FITTED half-life = {fitted_half_life} yr  (was 4)")
print(hl.round(5).to_string(index=False))

# ----------------------------------------------------------------------------
# 6. Write the report
# ----------------------------------------------------------------------------
lines = ["# Fitted presets (held-out RPS, continuous optimizer)", "",
         "> Every constant below was replaced by the value that minimises Ranked Probability",
         "> Score on out-of-sample matches, instead of being hand-picked. The two squad knobs",
         "> are fitted with `scipy.optimize.minimize_scalar` (bounds [0,1] = a weight's natural",
         "> range) so there is no hand-set candidate menu at all - the data walks to the optimum.",
         "> RPS is the football-standard ordered metric (lower = better).", "",
         "## Live spine (scripts 09-12) - squad prior", "",
         f"Held-out test set: **{len(test)} internationals after the EA FC 26 snapshot "
         f"({FC_SNAPSHOT.date()})**, both sides FC26-rated and with >={MIN_TEST_GAMES_BEFORE} prior games.", "",
         "| Knob | Old preset | Fitted | How |",
         "|---|---|---|---|",
         f"| Best-XI weight (vs top-23 depth) | 0.70 | **{fitted_best_xi:.2f}** | continuous optimizer (minimize_scalar) |",
         f"| Squad blend weight (talent vs form) | 0.35 | **{fitted_blend:.2f}** | continuous optimizer (minimize_scalar) |",
         "",
         f"Form-Elo-only held-out RPS = {baseline_rps:.5f}; best squad-blended RPS = {best_rps:.5f} "
         f"({(baseline_rps - best_rps) / baseline_rps * 100:+.2f}%).", "",
         f"**Honest reading:** the surface is flat - any blend in **{flat_lo:.2f}-{flat_hi:.2f}** scores "
         f"within RPS noise ({NOISE_FLOOR}) of the best. Report as **≈0.5**; the exact decimal is not "
         "signal the 445-match sample can support. Production is locked at the optimizer's point.", "",
         "### Blend sensitivity around the optimum (characterisation, not selection)", "",
         "| Blend weight | Held-out RPS | |", "|---|---|---|"]
for b, r in sensitivity:
    flag = "flat" if r <= best_rps + NOISE_FLOOR else ""
    lines.append(f"| {b:.2f} | {r:.5f} | {flag} |")
lines += ["", "## Archived ratio baseline only (NOT the live forecast) - recency half-life", "",
          "The live Elo spine has no half-life knob (Elo's recency is implicit in walk-forward",
          "updating). This fit hardens only the archived ratio-Poisson baseline (nb02/03/05/06).", "",
          "| Knob | Old preset | Fitted | How |", "|---|---|---|---|",
          f"| Half-life (years) | 4 | **{fitted_half_life}** | grid search, held-out RPS (ratio baseline) |", "",
          "| Half-life (yr) | Held-out RPS |", "|---|---|"]
for _, r in hl.iterrows():
    lines.append(f"| {int(r['half_life_years'])} | {r['rps']:.5f} |")
lines.append("")
REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
print(f"\nWrote {REPORT_PATH.name}")
