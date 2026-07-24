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

import importlib.util

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).parent
FIGURES = ROOT / "reports" / "figures"
# Raw slides -> gitignored scratch; 18_normalise_slides.py pads them into the final carousel_*.png.
RAW = FIGURES / "_raw"
RAW.mkdir(parents=True, exist_ok=True)

NAVY = {
    "ink": "#E6EAF2", "muted": "#96A1B8", "grid": "#2A3550", "surface": "#111A2B",
    "hit": "#4FBE8B", "miss": "#E87C82", "warm": "#E3B341", "row": "#18233A",
}

SLIDE = (10, 12.5)      # 4:5 portrait — the most screen a carousel gets on a phone
MARGIN = 0.9
CONTENT = 10 - 2 * MARGIN
CENTRE = 5.0

# ---------------------------------------------------------------- vertical spacing scale
# Three gaps, and only three. Grouping is carried by DISTANCE, so the sizes have to be clearly
# different or the reader cannot tell a new block from the next line of the same one: every step
# is roughly 1.7x the one below it. Earlier slides set each gap by eye, which is why the three
# bullets on the closing slide sat as far apart from each other as they did from the footer —
# they read as six loose items instead of one list of three.
#
#   TIGHT    consecutive lines of a single thought (a two-line heading, a label and its value)
#   GROUP    sibling items in one list — the rows of a table, the bullets of a set
#   SECTION  between blocks that are about different things
TIGHT, GROUP, SECTION = 0.62, 1.05, 1.80


def new_slide():
    fig, ax = plt.subplots(figsize=SLIDE)
    fig.patch.set_facecolor(NAVY["surface"])
    ax.set_facecolor(NAVY["surface"])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12.5)
    ax.axis("off")
    return fig, ax


def save(fig, name):
    out = RAW / name
    fig.savefig(out, dpi=160, facecolor=NAVY["surface"], bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def slide_hook():
    fig, ax = new_slide()
    ax.text(CENTRE, 9.42, "I forecast the", fontsize=27, color=NAVY["muted"],
            ha="center", va="center")
    # Full official name on the first mention (slide 3 then uses plain "the World Cup" in prose).
    # fontsize drops 40 -> 33 because the longer string would otherwise widen the tight bbox and
    # shrink the whole slide once 18_normalise_slides.py pads it to the 1600x2000 canvas.
    ax.text(CENTRE, 8.37, "FIFA World Cup 2026", fontsize=33, color=NAVY["ink"],
            ha="center", va="center", fontweight="bold")
    ax.text(CENTRE, 7.47, "before a ball was kicked.", fontsize=27, color=NAVY["muted"],
            ha="center", va="center")

    ax.plot([CENTRE - 1.3, CENTRE + 1.3], [6.77, 6.77], color=NAVY["hit"], linewidth=3.5)

    ax.text(CENTRE, 5.97, "Here is the report card —", fontsize=25, color=NAVY["ink"],
            ha="center", va="center")
    ax.text(CENTRE, 5.97 - TIGHT, "misses included.", fontsize=25, color=NAVY["warm"],
            ha="center", va="center", fontweight="bold")

    # Two lines rather than one: as a single line the method credit ran nearly the full width of
    # the slide, which made a footnote look like a headline.
    method_y = 5.97 - TIGHT - SECTION
    ax.text(CENTRE, method_y, "Elo-driven Dixon–Coles Poisson model",
            fontsize=14.5, color=NAVY["grid"], ha="center", va="center")
    ax.text(CENTRE, method_y - TIGHT, "10,000 simulations",
            fontsize=14.5, color=NAVY["grid"], ha="center", va="center")
    save(fig, "carousel_1_hook.png")


def coin_flip_ties():
    """Every knockout tie the model priced between 50% and 60%, from the locked ratings.

    Derived, not typed out. An earlier version hardcoded FOUR ties and claimed the model went
    2–2 on them; the band actually holds TEN and the model went 6–4. The four had been picked
    by hand and quietly excluded Colombia v Switzerland — the very tie slide 3 names. Reading
    the band off the same scorer slide 3 uses is what makes the two slides agree.

    Importing 15_calibration re-renders its own figure as a side effect. That is idempotent and
    cheap, and it is the price of having one scorer rather than two copies of the fixture list.
    """
    spec = importlib.util.spec_from_file_location("calibration", ROOT / "15_calibration.py")
    calibration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(calibration)

    scored = calibration.score_matches()
    band = scored[(scored.p_favourite >= 0.50) & (scored.p_favourite < 0.60)]
    band = band.sort_values("p_favourite")

    ties = []
    for tie in band.itertuples():
        underdog = tie.team_b if tie.favourite == tie.team_a else tie.team_a
        ties.append((f"{calibration.NAME[tie.favourite]} to beat {calibration.NAME[underdog]}",
                     f"{tie.p_favourite * 100:.0f}%", bool(tie.correct)))
    return ties


def slide_calibration():
    """Named ties instead of probability buckets — nothing for the reader to decode."""
    ties = coin_flip_ties()
    right = sum(1 for *_, correct in ties if correct)
    wrong = len(ties) - right

    fig, ax = new_slide()
    ax.text(CENTRE, 11.56, f"It called {len(ties)} ties", fontsize=31, color=NAVY["ink"],
            ha="center", va="center", fontweight="bold")
    ax.text(CENTRE, 11.56 - TIGHT, "too close to call", fontsize=31, color=NAVY["ink"],
            ha="center", va="center", fontweight="bold")
    ax.text(CENTRE, 11.56 - TIGHT - GROUP, "Barely better than a coin flip, said the model.",
            fontsize=16, color=NAVY["muted"], ha="center", va="center")

    # Ten rows is the most this slide can hold, so the row pitch is set from the box height plus
    # the smallest readable gap rather than from the scale — the rows are one list and want to
    # read as a solid block anyway.
    row_pitch = 0.61
    first_row = 11.56 - TIGHT - GROUP - SECTION
    for i, (text, pct, correct) in enumerate(ties):
        y = first_row - i * row_pitch
        tone = NAVY["hit"] if correct else NAVY["miss"]
        ax.add_patch(FancyBboxPatch((MARGIN, y - 0.25), CONTENT, 0.50,
                                    boxstyle="round,pad=0,rounding_size=0.10",
                                    facecolor=NAVY["row"], edgecolor=tone, linewidth=1.4))
        ax.text(MARGIN + 0.28, y, "✓" if correct else "✗", fontsize=13, color=tone,
                ha="left", va="center")
        ax.text(MARGIN + 0.72, y, text, fontsize=13.5, color=NAVY["ink"],
                ha="left", va="center")
        ax.text(10 - MARGIN - 0.28, y, pct, fontsize=13, color=NAVY["muted"],
                ha="right", va="center")

    verdict_y = first_row - (len(ties) - 1) * row_pitch - GROUP
    ax.text(CENTRE, verdict_y, f"{right} right. {wrong} wrong.", fontsize=25, color=NAVY["ink"],
            ha="center", va="center", fontweight="bold")
    ax.text(CENTRE, verdict_y - TIGHT, "About what a coin flip should look like.",
            fontsize=18, color=NAVY["warm"], ha="center", va="center")
    # "And the five it was most sure about? All five happened." was cut from here. Ten rows plus
    # a verdict already fill the slide, and that line is about the OTHER end of the confidence
    # range — a different claim wearing the same slide. It belongs in the post text.
    save(fig, "carousel_4_calibration.png")


def slide_close():
    fig, ax = new_slide()
    ax.text(CENTRE, 11.58, "What I'd change", fontsize=34, color=NAVY["ink"],
            ha="center", va="center", fontweight="bold")

    points = [
        ("Penalties", "Two of my four biggest misses went to a shootout.\n"
                      "A goals model has nothing to say about those."),
        ("Confidence", "It was consistently underconfident — the\n"
                       "probabilities were too timid, not too bold."),
        ("Draw luck", "Losing to the eventual champion isn't an error.\n"
                      "Judge a forecast by who beat you, not where you finished."),
    ]
    # One bullet spans a heading plus two body lines; BULLET_HEIGHT is that block, so the pitch
    # is height + GROUP. Before this the pitch was a flat 2.3, which left the same gap between
    # bullets as between the bullets and the footer — three items that belong together did not
    # look like they did.
    bullet_height = 1.25
    first_bullet = 9.78
    for i, (head, body) in enumerate(points):
        y = first_bullet - i * (bullet_height + GROUP)
        ax.plot([MARGIN, MARGIN], [y + 0.42, y - 1.05], color=NAVY["hit"], linewidth=3.5)
        ax.text(MARGIN + 0.45, y + 0.2, head, fontsize=23, color=NAVY["ink"],
                ha="left", va="center", fontweight="bold")
        ax.text(MARGIN + 0.45, y - 0.55, body, fontsize=15.2, color=NAVY["muted"],
                ha="left", va="center", linespacing=1.65)

    # These two lines were set in NAVY["grid"] — the gridline colour, not a text colour. On the
    # navy surface that is very nearly invisible, which is the wrong outcome for the claim the
    # whole deck rests on. They are the proof line; they get the body colour.
    last_bullet_bottom = first_bullet - (len(points) - 1) * (bullet_height + GROUP) - 1.05
    proof_y = last_bullet_bottom - SECTION
    ax.text(CENTRE, proof_y, "Every prediction was committed to git before kick-off.",
            fontsize=16, color=NAVY["muted"], ha="center", va="center")
    ax.text(CENTRE, proof_y - TIGHT, "The timestamps are the proof.",
            fontsize=16, color=NAVY["muted"], ha="center", va="center")

    # The "Code, data and every figure:" label was cut — a bare repo URL needs no introduction,
    # and the line it saves is what lets the block sit clear of the bottom edge. The link belongs
    # to the proof block above it, so the gap between them is GROUP, not SECTION.
    ax.text(CENTRE, proof_y - TIGHT - GROUP,
            "github.com/Ganapathy-K/fifa-world-cup-2026-forecast",
            fontsize=16, color=NAVY["hit"], ha="center", va="center", fontweight="bold")
    save(fig, "carousel_5_close.png")


slide_hook()
slide_calibration()
slide_close()
