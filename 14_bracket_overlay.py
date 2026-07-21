"""FIFA WC 2026 knockout bracket, mirrored like the official FIFA layout,
with the Poisson model's predictions overlaid as hit/miss on every team box.

Green + check = the model had this team reaching this round.
Red + cross   = it did not.
"""

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).parent
FIGURES = ROOT / "reports" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
FLAGS = ROOT / "assets" / "flags"

_flag_cache = {}


def flag(team):
    """Load a team's flag PNG once and reuse it; None if the file is missing."""
    if team not in _flag_cache:
        path = FLAGS / f"{team}.png"
        _flag_cache[team] = mpimg.imread(path) if path.exists() else None
    return _flag_cache[team]

# Dark mode is SELECTED, not an inverted copy: the fills are re-picked against the dark
# surface so the green/red split stays as legible as it is on white. Surface is a soft
# charcoal rather than pure black, and ink is off-white — pure #000/#FFF on a large
# figure is what makes dark mode tiring to read.
LIGHT = {
    "ink": "#1A1A1A", "muted": "#6B6B6B", "grid": "#D8D8D8", "surface": "#FFFFFF",
    "hit_fill": "#E3F1E4", "hit_edge": "#2E7D32",
    "miss_fill": "#FBE6E4", "miss_edge": "#C0392B",
    "na_fill": "#F2F2F2", "na_edge": "#B0B0B0",
}
DARK = {
    "ink": "#E7E9EC", "muted": "#9AA2AD", "grid": "#3A404A", "surface": "#191C21",
    "hit_fill": "#1B3325", "hit_edge": "#54B37A",
    "miss_fill": "#3A2220", "miss_edge": "#E0796C",
    "na_fill": "#24282E", "na_edge": "#5A616B",
}
# Navy variant — reads as "football tournament" without borrowing FIFA's actual marks.
# Slightly cooler than the charcoal, so the red misses sit a touch louder against it.
NAVY = {
    "ink": "#E6EAF2", "muted": "#96A1B8", "grid": "#33415C", "surface": "#111A2B",
    "hit_fill": "#16342C", "hit_edge": "#4FBE8B",
    "miss_fill": "#3A2028", "miss_edge": "#E87C82",
    "na_fill": "#1C2740", "na_edge": "#55637E",
}

# ---------------------------------------------------------------- actual results
# (team, goals, penalties or None, won)
L_R32 = [
    [("GER", 1, 3, False), ("PAR", 1, 4, True)],
    [("FRA", 3, None, True), ("SWE", 0, None, False)],
    [("RSA", 0, None, False), ("CAN", 1, None, True)],
    [("NED", 1, 2, False), ("MAR", 1, 3, True)],
    [("POR", 2, None, True), ("CRO", 1, None, False)],
    [("ESP", 3, None, True), ("AUT", 0, None, False)],
    [("USA", 2, None, True), ("BIH", 0, None, False)],
    [("BEL", 3, None, True), ("SEN", 2, None, False)],
]
R_R32 = [
    [("BRA", 2, None, True), ("JPN", 1, None, False)],
    [("CIV", 1, None, False), ("NOR", 2, None, True)],
    [("MEX", 2, None, True), ("ECU", 0, None, False)],
    [("ENG", 2, None, True), ("COD", 1, None, False)],
    [("ARG", 3, None, True), ("CPV", 2, None, False)],
    [("AUS", 1, 2, False), ("EGY", 1, 4, True)],
    [("SUI", 2, None, True), ("ALG", 0, None, False)],
    [("COL", 1, None, True), ("GHA", 0, None, False)],
]
L_R16 = [
    [("PAR", 0, None, False), ("FRA", 1, None, True)],
    [("CAN", 0, None, False), ("MAR", 3, None, True)],
    [("POR", 0, None, False), ("ESP", 1, None, True)],
    [("USA", 1, None, False), ("BEL", 4, None, True)],
]
R_R16 = [
    [("BRA", 1, None, False), ("NOR", 2, None, True)],
    [("MEX", 2, None, False), ("ENG", 3, None, True)],
    [("ARG", 3, None, True), ("EGY", 2, None, False)],
    [("SUI", 0, 4, True), ("COL", 0, 3, False)],
]
L_QF = [
    [("FRA", 2, None, True), ("MAR", 0, None, False)],
    [("ESP", 2, None, True), ("BEL", 1, None, False)],
]
R_QF = [
    [("NOR", 1, None, False), ("ENG", 2, None, True)],
    [("ARG", 3, None, True), ("SUI", 1, None, False)],
]
L_SF = [[("FRA", 0, None, False), ("ESP", 2, None, True)]]
R_SF = [[("ENG", 1, None, False), ("ARG", 2, None, True)]]
FINAL = [[("ESP", 1, None, True), ("ARG", 0, None, False)]]
BRONZE = [[("FRA", 4, None, False), ("ENG", 6, None, True)]]

# ------------------------------------------------- what the model predicted
# From MOST_LIKELY_BRACKET.md (chalk path). Teams the model had ALIVE at each round.
PRED = {
    "R32": {"GER", "PAR", "FRA", "SWE", "CZE", "CAN", "NED", "MAR", "COL", "CRO",
            "ESP", "AUT", "TUR", "BIH", "BEL", "KOR", "BRA", "JPN", "ECU", "NOR",
            "MEX", "SCO", "ENG", "SEN", "ARG", "URU", "USA", "IRN", "SUI", "EGY",
            "POR", "CIV"},
    "R16": {"GER", "FRA", "CAN", "NED", "COL", "ESP", "TUR", "BEL",
            "BRA", "NOR", "MEX", "ENG", "ARG", "USA", "SUI", "POR"},
    "QF": {"FRA", "NED", "ESP", "BEL", "BRA", "ENG", "ARG", "POR"},
    "SF": {"FRA", "ESP", "BRA", "ARG"},
    "F": {"ESP", "ARG"},
    # Third place: the model's chalk semis had France and Brazil losing, so those two
    # are its third-place match. Derived purely from pre-tournament ratings — France
    # 41.1% v Brazil 31.5% — so this is a genuine locked prediction, not hindsight.
    "BRONZE": {"FRA", "BRA"},
}

# Boxes are deliberately squat rather than wide: the score sits just right of the team
# code instead of being flushed to a far edge, which killed a lot of dead space per box
# and let the whole bracket shrink horizontally.
BOX_W, BOX_H, PAIR_GAP = 1.37, 0.50, 0.07


def draw_match(ax, x, y_center, match, round_key, out_of_scope=False):
    """Draw one two-team match box, colored by whether the model predicted each team here.

    out_of_scope renders neutral grey — for the third-place play-off, which the model
    never forecast at all, so neither a hit nor a miss.
    """
    for slot, (team, goals, pens, won) in enumerate(match):
        y = y_center + (PAIR_GAP / 2 + BOX_H / 2) * (1 if slot == 0 else -1)
        predicted = (not out_of_scope) and round_key is not None and team in PRED[round_key]
        if out_of_scope:
            fill, edge, mark = T["na_fill"], T["na_edge"], "–"
        elif predicted:
            fill, edge, mark = T["hit_fill"], T["hit_edge"], "✓"
        else:
            fill, edge, mark = T["miss_fill"], T["miss_edge"], "✗"

        ax.add_patch(FancyBboxPatch(
            (x, y - BOX_H / 2), BOX_W, BOX_H,
            boxstyle="round,pad=0,rounding_size=0.06",
            facecolor=fill, edgecolor=edge, linewidth=1.4, zorder=3,
        ))
        ax.text(x + 0.08, y, mark,
                fontsize=8.5, color=edge, va="center", ha="left", zorder=4)

        img = flag(team)
        if img is not None:
            ax.add_artist(AnnotationBbox(
                OffsetImage(img, zoom=0.19), (x + 0.36, y),
                frameon=False, box_alignment=(0.5, 0.5), zorder=4,
            ))

        ax.text(x + 0.56, y, team, fontsize=9.5, color=T["ink"], va="center", ha="left",
                fontweight="bold" if won else "normal", zorder=4)

        score = str(goals) if pens is None else f"{goals} ({pens})"
        ax.text(x + BOX_W - 0.08, y, score, fontsize=9,
                color=T["ink"] if won else T["muted"],
                va="center", ha="right", fontweight="bold" if won else "normal", zorder=4)


def connect(ax, x_from, y_from, x_to, y_to, rightward=True):
    """Elbow connector between a match and the next round."""
    x_mid = (x_from + x_to) / 2
    ax.plot([x_from, x_mid, x_mid, x_to], [y_from, y_from, y_to, y_to],
            color=T["grid"], linewidth=1.1, zorder=1, solid_joinstyle="miter")


Y_R32 = [15.4, 13.2, 11.0, 8.8, 6.6, 4.4, 2.2, 0.0]
Y_R16 = [14.3, 9.9, 5.5, 1.1]
Y_QF = [12.1, 3.3]
Y_SF = [7.7]
Y_FINAL, Y_BRONZE = 10.2, 4.6

XL = [0.3, 2.0, 3.7, 5.4]            # left columns: R32, R16, QF, SF
X_CENTER = 7.1
# Mirrored about the centre of the final box: x' = 2*(X_CENTER + BOX_W/2) - x - BOX_W.
XR = [round(2 * (X_CENTER + BOX_W / 2) - x - BOX_W, 2) for x in XL]


T = LIGHT  # active theme; build() swaps it


def build(bronze_neutral, out_name, theme=LIGHT):
    """Render the bracket. bronze_neutral greys the third-place box instead of marking it a miss."""
    global T
    T = theme
    fig, ax = plt.subplots(figsize=(17.5, 12))
    fig.patch.set_facecolor(T["surface"])
    ax.set_facecolor(T["surface"])

    for ys, x, data, key in [(Y_R32, XL[0], L_R32, "R32"), (Y_R16, XL[1], L_R16, "R16"),
                             (Y_QF, XL[2], L_QF, "QF"), (Y_SF, XL[3], L_SF, "SF")]:
        for y, match in zip(ys, data):
            draw_match(ax, x, y, match, key)

    for ys, x, data, key in [(Y_R32, XR[0], R_R32, "R32"), (Y_R16, XR[1], R_R16, "R16"),
                             (Y_QF, XR[2], R_QF, "QF"), (Y_SF, XR[3], R_SF, "SF")]:
        for y, match in zip(ys, data):
            draw_match(ax, x, y, match, key)

    draw_match(ax, X_CENTER, Y_FINAL, FINAL[0], "F")
    draw_match(ax, X_CENTER, Y_BRONZE, BRONZE[0], "BRONZE", out_of_scope=bronze_neutral)

    for i, y in enumerate(Y_R16):
        connect(ax, XL[0] + BOX_W, Y_R32[i * 2], XL[1], y)
        connect(ax, XL[0] + BOX_W, Y_R32[i * 2 + 1], XL[1], y)
        connect(ax, XR[0], Y_R32[i * 2], XR[1] + BOX_W, y)
        connect(ax, XR[0], Y_R32[i * 2 + 1], XR[1] + BOX_W, y)
    for i, y in enumerate(Y_QF):
        connect(ax, XL[1] + BOX_W, Y_R16[i * 2], XL[2], y)
        connect(ax, XL[1] + BOX_W, Y_R16[i * 2 + 1], XL[2], y)
        connect(ax, XR[1], Y_R16[i * 2], XR[2] + BOX_W, y)
        connect(ax, XR[1], Y_R16[i * 2 + 1], XR[2] + BOX_W, y)
    for y in Y_QF:
        connect(ax, XL[2] + BOX_W, y, XL[3], Y_SF[0])
        connect(ax, XR[2], y, XR[3] + BOX_W, Y_SF[0])
    connect(ax, XL[3] + BOX_W, Y_SF[0], X_CENTER, Y_FINAL)
    connect(ax, XR[3], Y_SF[0], X_CENTER + BOX_W, Y_FINAL)

    headers = [("Round of 32", XL[0]), ("Round of 16", XL[1]), ("Quarter-final", XL[2]),
               ("Semi-final", XL[3]), ("Semi-final", XR[3]), ("Quarter-final", XR[2]),
               ("Round of 16", XR[1]), ("Round of 32", XR[0])]
    for label, x in headers:
        ax.text(x + BOX_W / 2, 16.5, label, fontsize=10, color=T["muted"], ha="center", va="center")

    ax.text(X_CENTER + BOX_W / 2, Y_FINAL + 0.95, "FINAL", fontsize=11.5, color=T["ink"],
            ha="center", va="center", fontweight="bold")
    bronze_label = ("Third place  (not forecast by the model)" if bronze_neutral
                    else "Third place   ·   model picked France to win it")
    ax.text(X_CENTER + BOX_W / 2, Y_BRONZE + 0.95, bronze_label, fontsize=9.5, color=T["muted"],
            ha="center", va="center")

    ax.text(0.3, 18.0, "FIFA World Cup 2026 — what my model got right",
            fontsize=20, color=T["ink"], ha="left", va="center", fontweight="bold")
    ax.text(0.3, 17.35,
            "Poisson + Elo model, predictions locked before a ball was kicked. "
            "Every team box marked against what the model predicted for that round.",
            fontsize=11, color=T["muted"], ha="left", va="center")

    ax.add_patch(FancyBboxPatch((0.3, 16.75), 0.28, 0.28, boxstyle="round,pad=0,rounding_size=0.05",
                                facecolor=T["hit_fill"], edgecolor=T["hit_edge"], linewidth=1.4))
    ax.text(0.72, 16.89, "✓  model had this team reaching this round", fontsize=9.5,
            color=T["muted"], va="center", ha="left")
    ax.add_patch(FancyBboxPatch((4.15, 16.75), 0.28, 0.28, boxstyle="round,pad=0,rounding_size=0.05",
                                facecolor=T["miss_fill"], edgecolor=T["miss_edge"], linewidth=1.4))
    ax.text(4.57, 16.89, "✗  it did not", fontsize=9.5, color=T["muted"], va="center", ha="left")
    if bronze_neutral:
        ax.add_patch(FancyBboxPatch((6.1, 16.75), 0.28, 0.28,
                                    boxstyle="round,pad=0,rounding_size=0.05",
                                    facecolor=T["na_fill"], edgecolor=T["na_edge"], linewidth=1.4))
        ax.text(6.52, 16.89, "–  not forecast", fontsize=9.5, color=T["muted"],
                va="center", ha="left")

    # Ordered by the date each was decided — third place (Jul 18) precedes the final (Jul 19).
    tally = ("Last 32: 26/32   ·   Last 16: 13/16   ·   Last 8: 5/8   ·   Last 4: 3/4   ·   "
             "Final: 2/2")
    if not bronze_neutral:
        tally += "   ·   Third place: England ✗"
    tally += "   ·   Champion: Spain ✓"
    ax.text(X_CENTER + BOX_W / 2, -1.35, tally,
            fontsize=12.5, color=T["ink"], ha="center", va="center", fontweight="bold")

    ax.set_xlim(-0.2, XR[0] + BOX_W + 0.2)
    ax.set_ylim(-1.8, 18.5)
    ax.axis("off")
    plt.tight_layout()
    out = FIGURES / out_name
    plt.savefig(out, dpi=160, facecolor=T["surface"], bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


build(bronze_neutral=False, out_name="wc2026_bracket.png", theme=LIGHT)
build(bronze_neutral=False, out_name="wc2026_bracket_dark.png", theme=DARK)
build(bronze_neutral=False, out_name="wc2026_bracket_navy.png", theme=NAVY)
build(bronze_neutral=True, out_name="wc2026_bracket_bronze_greyed.png", theme=LIGHT)
