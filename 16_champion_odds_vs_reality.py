"""Pre-tournament title odds against how each team's tournament actually ended.

Source of odds: CHAMPION_ODDS_FINAL.md — the canonical forecast (raw Elo + confederation
calibration + FC26 squad-strength prior, 10,000 sims). The 26.5% figures in
CHAMPION_ODDS_ADJUSTED.md are an intermediate stage, not the final.

Why teams are judged by their ELIMINATOR rather than by their finishing position:
the 10k simulation already prices in the bracket. France's 17.5% is what it is *because*
France might meet Spain before the final — which is exactly what happened. Grading France
as "fell short" for finishing 4th would punish it twice for a draw the model already knew
about. Two teams in the same half cannot both reach the final.

So the only meaningful question is: were you beaten by a favourite, or by an underdog?

On wording: "favourite / underdog" rather than "stronger / weaker". Stronger-vs-weaker makes a
claim about the teams, which this chart cannot support — Norway beat Brazil, so which of them
was stronger? Favourite-vs-underdog makes a claim about the FORECAST, which is precisely what
is being measured, and the two terms are a natural pair.
"""

from pathlib import Path

import math

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Circle, Polygon, Rectangle

ROOT = Path(__file__).parent
FIGURES = ROOT / "reports" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
FLAGS = ROOT / "assets" / "flags"

CODE = {
    "Spain": "ESP", "France": "FRA", "Argentina": "ARG", "Brazil": "BRA", "England": "ENG",
    "Portugal": "POR", "Germany": "GER", "Belgium": "BEL", "Netherlands": "NED",
    "Colombia": "COL", "Morocco": "MAR", "Switzerland": "SUI", "Croatia": "CRO",
    "Uruguay": "URU", "Turkey": "TUR", "Norway": "NOR", "Paraguay": "PAR",
}

NAVY = {
    "ink": "#E6EAF2", "muted": "#96A1B8", "grid": "#2A3550", "surface": "#111A2B",
    "champion": "#E3B341", "expected": "#5B7FB5", "upset": "#E87C82",
    "cointoss": "#77839B",
}

# How each knockout exit was decided, and what the model gave the favourite IN THAT MATCH.
# Title odds are the wrong yardstick for "was this an upset": Morocco's 1.5% chance of
# winning the tournament reflects a long path, not weakness. For a single tie the match
# probability is the honest measure — and by it, Netherlands v Morocco (57%) and
# Colombia v Switzerland (54%) were toss-ups the model never claimed to call.
MATCH_ODDS = {"Germany": 76, "Brazil": 69, "Netherlands": 57, "Colombia": 54}
COIN_FLIP_MAX = 60          # at or below this, the model expressed no real opinion

# Shootout exits, written from the losing team's point of view. Carrying the SCORE rather
# than an icon is the football convention, and it adds information instead of decorating:
# an icon can only say "something happened here", the score says what.
ON_PENS = {"Germany": "3–4", "Netherlands": "2–3", "Colombia": "3–4"}

# team -> pre-tournament champion %. Teams outside the published top 20 were all under 0.3%.
ODDS = {
    "Spain": 20.2, "France": 17.5, "Argentina": 13.7, "Brazil": 11.2, "England": 9.4,
    "Portugal": 7.0, "Germany": 5.9, "Belgium": 3.0, "Netherlands": 2.9, "Colombia": 1.8,
    "Morocco": 1.5, "Switzerland": 0.9, "Croatia": 0.9, "Norway": 0.6,
}
# Uruguay (0.8%) and Turkey (0.7%) are dropped: both exited in the group stage, so neither
# has an eliminator and neither carries a story. Their rows only cost vertical space.
LONGSHOT = 0.3  # anyone unlisted

# team -> (how their run ended, who ended it or None)
# Stages use the standard short form — R32 · R16 · QF · SF · F. Safe here because slide 2
# (the bracket) spells every stage out as a column header, so the deck teaches the vocabulary
# before this slide reuses it. Abbreviate all or none: "Semi-final" sitting next to "R16"
# reads as sloppiness rather than as a choice.
ENDINGS = {
    "Spain": ("Won the World Cup", None),
    "France": ("SF", "Spain"),
    # "Final" is left long: a lone "F" reads as a grade or a typo, and it's already the
    # shortest stage name, so the abbreviation bought nothing.
    "Argentina": ("Final", "Spain"),
    "Brazil": ("R16", "Norway"),
    "England": ("SF", "Argentina"),
    "Portugal": ("R16", "Spain"),
    "Germany": ("R32", "Paraguay"),
    "Belgium": ("QF", "Spain"),
    "Netherlands": ("R32", "Morocco"),
    "Colombia": ("R16", "Switzerland"),
    "Morocco": ("QF", "France"),
    "Switzerland": ("QF", "Argentina"),
    "Croatia": ("R32", "Portugal"),
    "Norway": ("QF", "England"),
}


def classify(team):
    """Champion, beaten by a favourite, lost a toss-up, or genuinely upset.

    Classification uses the MATCH probability, not the title odds — see MATCH_ODDS.
    """
    stage, beaten_by = ENDINGS[team]
    if beaten_by is None:
        return "champion"
    if ODDS.get(beaten_by, LONGSHOT) >= ODDS[team]:
        return "expected"
    return "cointoss" if MATCH_ODDS.get(team, 100) <= COIN_FLIP_MAX else "upset"


_flags = {}


def flag(team):
    """Flag image for a team, or None when the file is missing."""
    code = CODE.get(team)
    if code not in _flags:
        path = FLAGS / f"{code}.png"
        _flags[code] = mpimg.imread(path) if path.exists() else None
    return _flags[code]


teams = sorted(ODDS, key=ODDS.get, reverse=True)


BALL = ROOT / "assets" / "football.png"


def make_football(path, size=200):
    """Generate a football icon once, as a PNG.

    Drawn rather than downloaded: no licence question, no network dependency, and it renders
    at whatever resolution we ask for. A circle in DATA coordinates would distort, since the
    x and y scales here differ — an image does not, which is the same reason the flags work.
    """
    fig = plt.figure(figsize=(size / 100, size / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_alpha(0)

    ax.add_patch(Circle((0, 0), 1.0, facecolor="white", edgecolor="#1A1A1A", linewidth=7))

    def pentagon(cx, cy, r, rotation):
        return [(cx + r * math.cos(math.radians(rotation + k * 72)),
                 cy + r * math.sin(math.radians(rotation + k * 72))) for k in range(5)]

    # One bold central pentagon plus five chunky rim patches. Thin seams read as a gear at
    # icon size; solid shapes are what makes it recognisably a football.
    ax.add_patch(Polygon(pentagon(0, 0, 0.46, 90), closed=True, facecolor="#1A1A1A"))
    for i in range(5):
        angle = math.radians(90 + i * 72 + 36)
        ax.add_patch(Polygon(pentagon(0.92 * math.cos(angle), 0.92 * math.sin(angle),
                                      0.34, math.degrees(angle) + 180),
                             closed=True, facecolor="#1A1A1A",
                             clip_path=Circle((0, 0), 0.96, transform=ax.transData)))

    fig.savefig(path, dpi=100, transparent=True)
    plt.close(fig)


if not BALL.exists():
    BALL.parent.mkdir(parents=True, exist_ok=True)
    make_football(BALL)
    print(f"generated {BALL}")

_ball_img = mpimg.imread(BALL)


def draw_ball(ax, x, y):
    """Mark a tie that was decided by a penalty shootout."""
    ax.add_artist(AnnotationBbox(OffsetImage(_ball_img, zoom=0.085), (x, y),
                                 frameon=False, box_alignment=(0.5, 0.5), zorder=5))


def build(show_ball, out_name, eliminator_flags=False):
    fig, ax = plt.subplots(figsize=(13, 9))
    fig.patch.set_facecolor(NAVY["surface"])
    ax.set_facecolor(NAVY["surface"])
    pending = []

    for i, team in enumerate(teams):
        stage, beaten_by = ENDINGS[team]
        tone = classify(team)
        ax.barh(i, ODDS[team], height=0.62, color=NAVY[tone], alpha=0.92, zorder=3)

        # Flag and code are drawn FIRST, before any early exit below — otherwise the
        # champion row (which returns early) loses its identity entirely.
        img = flag(team)
        if img is not None:
            ax.add_artist(AnnotationBbox(OffsetImage(img, zoom=0.175), (-2.15, i),
                                         frameon=False, box_alignment=(0.5, 0.5),
                                         annotation_clip=False, zorder=4))
        ax.text(-1.62, i, CODE[team], fontsize=12, color=NAVY["ink"], fontweight="bold",
                ha="left", va="center", clip_on=False, zorder=4)

        # The ball marks a shootout. Grey already carries "the model called it a toss-up",
        # so the ball is free to carry the orthogonal fact: how the tie was settled.
        text_x = ODDS[team] + 0.5
        if show_ball and team in ON_PENS:
            draw_ball(ax, ODDS[team] + 0.62, i)
            text_x = ODDS[team] + 1.35

        colour = NAVY[tone] if tone in ("champion", "upset") else NAVY["muted"]
        weight = "bold" if tone in ("champion", "upset") else "normal"

        if beaten_by is None:
            label = stage.upper() if tone == "champion" else stage
            ax.text(text_x, i, label, va="center", ha="left", fontsize=11,
                    color=colour, fontweight=weight, zorder=4)
            continue

        eliminator_odds = ODDS.get(beaten_by)
        odds_text = f"{eliminator_odds}%" if eliminator_odds else "<0.3%"
        code = CODE.get(beaten_by, beaten_by)
        tail = f"{code} ({odds_text})"
        if team in ON_PENS:
            tail += f", {ON_PENS[team]} on pens"

        if not eliminator_flags:
            ax.text(text_x, i, f"{stage} — out to {tail}", va="center", ha="left",
                    fontsize=11, color=colour, fontweight=weight, zorder=4)
            continue

        # Split the label so the eliminator's flag can sit between the prefix and the code.
        # The prefix width is measured from the rendered text rather than guessed from a
        # character count, so the flag lands correctly whatever the stage name's length.
        prefix = ax.text(text_x, i, f"{stage} — out to ", va="center", ha="left",
                         fontsize=11, color=colour, fontweight=weight, zorder=4)
        pending.append((prefix, beaten_by, tail, i, colour, weight))

    ax.set_yticks(range(len(teams)))
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    ax.set_xlim(0, 34)

    if pending:
        # Measure the drawn prefixes, then drop each eliminator's flag at the end of its
        # prefix and continue the label after it.
        fig.canvas.draw()
        inv = ax.transData.inverted()
        for prefix, beaten_by, tail, row, colour, weight in pending:
            end_x = inv.transform((prefix.get_window_extent().x1, 0))[0]
            img = flag(beaten_by)
            if img is not None:
                ax.add_artist(AnnotationBbox(OffsetImage(img, zoom=0.155),
                                             (end_x + 0.32, row), frameon=False,
                                             box_alignment=(0.5, 0.5),
                                             annotation_clip=False, zorder=4))
                end_x += 0.72
            ax.text(end_x, row, tail, va="center", ha="left", fontsize=11,
                    color=colour, fontweight=weight, zorder=4)
    ax.set_xlabel("pre-tournament chance of winning the World Cup (%)",
                  fontsize=11.5, color=NAVY["muted"], labelpad=12)
    ax.tick_params(colors=NAVY["muted"], labelsize=10)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(NAVY["grid"])
    ax.grid(axis="x", color=NAVY["grid"], linewidth=0.8, alpha=0.55)
    ax.set_axisbelow(True)

    ax.set_title("Beaten by a favourite — or by an underdog?",
                 fontsize=19, color=NAVY["ink"], fontweight="bold", loc="left", pad=52)
    ax.text(0, 1.028,
            "Bars = title chance over 10,000 pre-tournament simulations. "
            "Labels = how the run ended, and who ended it.",
            transform=ax.transAxes, fontsize=11, color=NAVY["muted"], va="bottom")

    # Caption and legend both live INSIDE the axes, bottom-right. Because the bars are
    # sorted by descending odds, the lower-right is empty by construction — so the figure
    # needs no strip beneath the axis at all.
    # Vertical rhythm: a big gap ABOVE the block and an equal one BELOW it (down to the
    # x-axis), with smaller equal gaps between the three blocks. That makes the furniture
    # read as one group floating in the empty wedge, rather than as three loose items.
    note_x = 16.4
    ax.text(note_x, 7.08,
            "Losing to the eventual champion is not a modelling error —\n"
            "the simulation already priced in the bracket.",
            fontsize=10.5, color=NAVY["muted"], va="top", ha="left", linespacing=1.7)
    ax.text(note_x, 8.57,
            "Only two exits were true upsets — the model rated Netherlands'\n"
            "and Colombia's ties at just 57% and 54%. All three shootouts\n"
            "were lost by a single penalty, which no goals model forecasts.",
            fontsize=10.5, color=NAVY["muted"], va="top", ha="left", linespacing=1.7)

    # "too close to call" over "toss-up" or "coin flip": it needs no prior knowledge, and it
    # is already the title of slide 4 — so the deck says one thing twice rather than two
    # things once. "Coin flip" still earns its place in slide 4's body as the explanation.
    legend = [("won it", "champion"), ("beaten by a favourite", "expected"),
              ("too close to call", "cointoss"), ("beaten by an underdog", "upset")]
    for j, (label, tone) in enumerate(legend):
        y = 10.66 + j * 0.64
        ax.add_patch(Rectangle((note_x, y - 0.19), 0.52, 0.38,
                               facecolor=NAVY[tone], zorder=5))
        ax.text(note_x + 0.95, y, label, fontsize=10.5, color=NAVY["muted"],
                va="center", ha="left")

    out = FIGURES / out_name
    plt.savefig(out, dpi=160, facecolor=NAVY["surface"], bbox_inches="tight",
                pad_inches=0.55)
    plt.close(fig)
    print(f"saved {out}")


# Ball is the default: it catches the eye and pulls you to the row, the score explains why
# once you are there. The text-only variant stays available for comparison.
build(show_ball=True, out_name="wc2026_champion_odds_vs_reality.png",
      eliminator_flags=True)
build(show_ball=False, out_name="wc2026_champion_odds_vs_reality_noball.png",
      eliminator_flags=True)
for team in teams:
    print(f"  {team:<13} {ODDS[team]:>5}%  {classify(team)}")
