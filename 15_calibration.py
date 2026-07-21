"""Reliability (calibration) diagram for the 32 knockout matches of WC 2026.

Accuracy asks "was it right?". Calibration asks "did it know how sure to be?" —
when the model said 70%, did that happen ~70% of the time?

Every match is re-scored with the LOCKED pre-tournament ratings, on the matchups
that actually occurred. Knockouts must produce a winner, so the draw probability
is split evenly between the two sides.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from match_engine import match_outcome_probabilities

ROOT = Path(__file__).parent
FIGURES = ROOT / "reports" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

NAVY = {
    "ink": "#E6EAF2", "muted": "#96A1B8", "grid": "#2A3550", "surface": "#111A2B",
    "dot": "#4FBE8B", "reference": "#6E7C99", "warn": "#E87C82",
}

NAME = {
    "GER": "Germany", "PAR": "Paraguay", "FRA": "France", "SWE": "Sweden",
    "RSA": "South Africa", "CAN": "Canada", "NED": "Netherlands", "MAR": "Morocco",
    "POR": "Portugal", "CRO": "Croatia", "ESP": "Spain", "AUT": "Austria",
    "USA": "United States", "BIH": "Bosnia and Herzegovina", "BEL": "Belgium",
    "SEN": "Senegal", "BRA": "Brazil", "JPN": "Japan", "CIV": "Ivory Coast",
    "NOR": "Norway", "MEX": "Mexico", "ECU": "Ecuador", "ENG": "England",
    "COD": "DR Congo", "ARG": "Argentina", "CPV": "Cape Verde", "AUS": "Australia",
    "EGY": "Egypt", "SUI": "Switzerland", "ALG": "Algeria", "COL": "Colombia",
    "GHA": "Ghana",
}

# (team A, team B, who actually advanced) — all 32 knockout ties, in order.
MATCHES = [
    ("GER", "PAR", "PAR"), ("FRA", "SWE", "FRA"), ("RSA", "CAN", "CAN"), ("NED", "MAR", "MAR"),
    ("POR", "CRO", "POR"), ("ESP", "AUT", "ESP"), ("USA", "BIH", "USA"), ("BEL", "SEN", "BEL"),
    ("BRA", "JPN", "BRA"), ("CIV", "NOR", "NOR"), ("MEX", "ECU", "MEX"), ("ENG", "COD", "ENG"),
    ("ARG", "CPV", "ARG"), ("AUS", "EGY", "EGY"), ("SUI", "ALG", "SUI"), ("COL", "GHA", "COL"),
    ("PAR", "FRA", "FRA"), ("CAN", "MAR", "MAR"), ("POR", "ESP", "ESP"), ("USA", "BEL", "BEL"),
    ("BRA", "NOR", "NOR"), ("MEX", "ENG", "ENG"), ("ARG", "EGY", "ARG"), ("SUI", "COL", "SUI"),
    ("FRA", "MAR", "FRA"), ("ESP", "BEL", "ESP"), ("NOR", "ENG", "ENG"), ("ARG", "SUI", "ARG"),
    ("FRA", "ESP", "ESP"), ("ENG", "ARG", "ARG"), ("FRA", "ENG", "ENG"), ("ESP", "ARG", "ESP"),
]

BUCKETS = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 1.001)]


def score_matches():
    """Re-run every actual knockout tie through the locked pre-tournament model."""
    processed = ROOT / "data" / "processed"
    ratings = dict(pd.read_parquet(processed / "wc_final_ratings.parquet")
                   .itertuples(index=False, name=None))
    params = pd.read_parquet(processed / "supremacy_params.parquet").iloc[0]
    slope, total_goals = params["slope"], params["total_goals"]

    rows = []
    for team_a, team_b, advanced in MATCHES:
        supremacy = slope * (ratings[NAME[team_a]] - ratings[NAME[team_b]])
        expected_a = max((total_goals + supremacy) / 2, 0.05)
        expected_b = max((total_goals - supremacy) / 2, 0.05)
        outcome = match_outcome_probabilities(expected_a, expected_b)
        prob_a = outcome["home_win"] + outcome["draw"] / 2
        favourite, prob = (team_a, prob_a) if prob_a >= 0.5 else (team_b, 1 - prob_a)
        rows.append({
            "team_a": team_a, "team_b": team_b, "favourite": favourite,
            "p_favourite": prob, "advanced": advanced, "correct": favourite == advanced,
        })
    return pd.DataFrame(rows)


def bucket_table(df):
    """Group predictions by confidence and compare claimed vs observed hit rate."""
    rows = []
    for low, high in BUCKETS:
        slice_ = df[(df.p_favourite >= low) & (df.p_favourite < high)]
        if len(slice_):
            rows.append({
                "label": f"{int(low * 100)}–{min(int(high * 100), 100)}%",
                "n": len(slice_),
                "claimed": slice_.p_favourite.mean() * 100,
                "observed": slice_.correct.mean() * 100,
            })
    return pd.DataFrame(rows)


df = score_matches()
buckets = bucket_table(df)
df.to_csv(ROOT / "reports" / "calibration_knockouts.csv", index=False)

fig, ax = plt.subplots(figsize=(9.5, 9))
fig.patch.set_facecolor(NAVY["surface"])
ax.set_facecolor(NAVY["surface"])

ax.plot([50, 100], [50, 100], linestyle=(0, (4, 4)), linewidth=1.6,
        color=NAVY["reference"], zorder=2)
ax.text(88, 84, "perfectly honest", fontsize=10, color=NAVY["reference"],
        rotation=45, rotation_mode="anchor", ha="center", va="center")
ax.text(56, 96, "was RIGHT more often than it claimed\n(underconfident)",
        fontsize=10.5, color=NAVY["dot"], ha="left", va="top", linespacing=1.5)
ax.text(80, 56, "talked bigger than it delivered\n(overconfident)",
        fontsize=10.5, color=NAVY["warn"], ha="left", va="top", linespacing=1.5)

for _, row in buckets.iterrows():
    ax.plot([row.claimed, row.claimed], [row.claimed, row.observed],
            color=NAVY["dot"], linewidth=1.4, alpha=0.45, zorder=3)
    ax.scatter(row.claimed, row.observed, s=90 + row.n * 34, color=NAVY["dot"],
               edgecolor=NAVY["surface"], linewidth=2.2, zorder=4)
    ax.annotate(f"{row.label}\n{row.n} matches", (row.claimed, row.observed),
                textcoords="offset points", xytext=(0, -34), ha="center",
                fontsize=9.5, color=NAVY["muted"], linespacing=1.4)

ax.set_xlim(48, 102)
ax.set_ylim(40, 108)
ax.set_xticks([50, 60, 70, 80, 90, 100])
ax.set_yticks([50, 60, 70, 80, 90, 100])
ax.set_xlabel("what the model claimed", fontsize=11.5, color=NAVY["muted"], labelpad=12)
ax.set_ylabel("what actually happened", fontsize=11.5, color=NAVY["muted"], labelpad=12)
ax.tick_params(colors=NAVY["muted"], labelsize=10)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(NAVY["grid"])
ax.grid(color=NAVY["grid"], linewidth=0.8, alpha=0.55)
ax.set_axisbelow(True)

hit_rate = df.correct.mean() * 100
ax.set_title("Did the model know how sure to be?", fontsize=19, color=NAVY["ink"],
             fontweight="bold", loc="left", pad=56)
ax.text(0, 1.022,
        f"All {len(df)} knockout ties, re-scored with the locked pre-tournament ratings. "
        f"The favourite won {df.correct.sum()} of {len(df)} ({hit_rate:.0f}%).",
        transform=ax.transAxes, fontsize=11, color=NAVY["muted"], va="bottom")
ax.text(0, -0.115,
        "Every point sits above the line: the model was consistently MORE right than it "
        "claimed to be — underconfident, not overconfident.\n"
        "Caveats: neutral venue assumed throughout (no host bump), and 5–10 matches per "
        "bucket is a thin sample.",
        transform=ax.transAxes, fontsize=10, color=NAVY["muted"], va="top", linespacing=1.6)

plt.tight_layout()
out = FIGURES / "wc2026_calibration.png"
plt.savefig(out, dpi=160, facecolor=NAVY["surface"], bbox_inches="tight")
print(buckets.to_string(index=False))
print(f"\nsaved {out}")
