"""Opening, calibration and closing slides for the WC-2026 report-card carousel.

The three data figures (bracket, odds-vs-reality, calibration diagram) come from scripts
14–16. This builds the slides that carry the narrative, in the same navy palette so the
deck reads as one object.

Two deliberate choices:

1. Everything sits inside ONE content column (MARGIN..10-MARGIN) with a shared vertical
   rhythm. Earlier versions set x by eye per element, so titles overhung the right edge
   while body text sat left and nothing aligned.
2. The calibration slide shows FOUR NAMED MATCHES rather than probability buckets. "It
   said 50-60% sure and went 60%" asks a reader to decode two abstractions; "it called
   these four coin flips and went 2-2" is the same finding with nothing to decode. It also
   reuses the green tick / red cross language from the bracket slide, so the deck teaches
   its own vocabulary once and then reuses it.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).parent
FIGURES = ROOT / "reports" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

NAVY = {
    "ink": "#E6EAF2", "muted": "#96A1B8", "grid": "#2A3550", "surface": "#111A2B",
    "hit": "#4FBE8B", "miss": "#E87C82", "warm": "#E3B341", "row": "#18233A",
}

SLIDE = (10, 12.5)      # 4:5 portrait — the most screen a carousel gets on a phone
MARGIN = 0.9
CONTENT = 10 - 2 * MARGIN
CENTRE = 5.0


def new_slide():
    fig, ax = plt.subplots(figsize=SLIDE)
    fig.patch.set_facecolor(NAVY["surface"])
    ax.set_facecolor(NAVY["surface"])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12.5)
    ax.axis("off")
    return fig, ax


def save(fig, name):
    out = FIGURES / name
    fig.savefig(out, dpi=160, facecolor=NAVY["surface"], bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def slide_hook():
    fig, ax = new_slide()
    ax.text(CENTRE, 10.9, "I forecast the", fontsize=27, color=NAVY["muted"],
            ha="center", va="center")
    ax.text(CENTRE, 9.85, "2026 World Cup", fontsize=40, color=NAVY["ink"],
            ha="center", va="center", fontweight="bold")
    ax.text(CENTRE, 8.95, "before a ball was kicked.", fontsize=27, color=NAVY["muted"],
            ha="center", va="center")

    ax.plot([CENTRE - 1.3, CENTRE + 1.3], [8.25, 8.25], color=NAVY["hit"], linewidth=3.5)

    ax.text(CENTRE, 7.45, "Here is the report card —", fontsize=25, color=NAVY["ink"],
            ha="center", va="center")
    ax.text(CENTRE, 6.72, "misses included.", fontsize=25, color=NAVY["warm"],
            ha="center", va="center", fontweight="bold")

    lines = [("It called the champion.", NAVY["hit"]),
             ("It called the exact final.", NAVY["hit"]),
             ("It also backed Brazil, Germany and Portugal.", NAVY["miss"])]
    for i, (line, colour) in enumerate(lines):
        y = 5.15 - i * 0.82
        ax.text(CENTRE, y, line, fontsize=19, color=colour, ha="center", va="center")

    ax.text(CENTRE, 2.0, "Elo-driven Dixon–Coles Poisson model  ·  10,000 simulations",
            fontsize=14.5, color=NAVY["grid"], ha="center", va="center")
    save(fig, "carousel_1_hook.png")


def slide_calibration():
    """Four named ties instead of probability buckets — nothing for the reader to decode."""
    fig, ax = new_slide()
    ax.text(CENTRE, 11.5, "It called four ties", fontsize=33, color=NAVY["ink"],
            ha="center", va="center", fontweight="bold")
    ax.text(CENTRE, 10.75, "too close to call", fontsize=33, color=NAVY["ink"],
            ha="center", va="center", fontweight="bold")
    ax.text(CENTRE, 9.85, "Barely better than a coin flip, said the model.",
            fontsize=18, color=NAVY["muted"], ha="center", va="center")

    ties = [("Ecuador to beat Mexico", "54%", False),
            ("Egypt to beat Australia", "55%", True),
            ("Netherlands to beat Morocco", "57%", False),
            ("Switzerland to beat Algeria", "57%", True)]

    for i, (text, pct, correct) in enumerate(ties):
        y = 8.35 - i * 1.15
        tone = NAVY["hit"] if correct else NAVY["miss"]
        ax.add_patch(FancyBboxPatch((MARGIN, y - 0.42), CONTENT, 0.84,
                                    boxstyle="round,pad=0,rounding_size=0.12",
                                    facecolor=NAVY["row"], edgecolor=tone, linewidth=1.6))
        ax.text(MARGIN + 0.35, y, "✓" if correct else "✗", fontsize=18, color=tone,
                ha="left", va="center")
        ax.text(MARGIN + 0.85, y, text, fontsize=18, color=NAVY["ink"],
                ha="left", va="center")
        ax.text(10 - MARGIN - 0.35, y, pct, fontsize=17, color=NAVY["muted"],
                ha="right", va="center")

    ax.text(CENTRE, 3.35, "Two right. Two wrong.", fontsize=26, color=NAVY["ink"],
            ha="center", va="center", fontweight="bold")
    ax.text(CENTRE, 2.6, "Exactly what a coin flip should look like.",
            fontsize=20, color=NAVY["warm"], ha="center", va="center")
    ax.text(CENTRE, 1.35,
            "And the five it was most sure about? All five happened.",
            fontsize=17, color=NAVY["muted"], ha="center", va="center")
    save(fig, "carousel_4_calibration.png")


def slide_close():
    fig, ax = new_slide()
    ax.text(CENTRE, 11.4, "What I'd change", fontsize=34, color=NAVY["ink"],
            ha="center", va="center", fontweight="bold")

    points = [
        ("Penalties", "Two of my four biggest misses went to a shootout.\n"
                      "A goals model has nothing to say about those."),
        ("Confidence", "It was consistently underconfident — the\n"
                       "probabilities were too timid, not too bold."),
        ("Draw luck", "Losing to the eventual champion isn't an error.\n"
                      "Judge a forecast by who beat you, not where you finished."),
    ]
    for i, (head, body) in enumerate(points):
        y = 9.4 - i * 2.3
        ax.plot([MARGIN, MARGIN], [y + 0.42, y - 1.05], color=NAVY["hit"], linewidth=3.5)
        ax.text(MARGIN + 0.45, y + 0.2, head, fontsize=23, color=NAVY["ink"],
                ha="left", va="center", fontweight="bold")
        ax.text(MARGIN + 0.45, y - 0.55, body, fontsize=16.5, color=NAVY["muted"],
                ha="left", va="center", linespacing=1.65)

    ax.text(CENTRE, 2.35, "Every prediction was committed to git before kick-off.",
            fontsize=16, color=NAVY["grid"], ha="center", va="center")
    ax.text(CENTRE, 1.85, "The timestamps are the proof.",
            fontsize=16, color=NAVY["grid"], ha="center", va="center")
    ax.text(CENTRE, 0.95, "Code, data and every figure:  github.com/<your-handle>",
            fontsize=17, color=NAVY["hit"], ha="center", va="center", fontweight="bold")
    save(fig, "carousel_5_close.png")


slide_hook()
slide_calibration()
slide_close()
