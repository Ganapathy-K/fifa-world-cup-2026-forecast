"""Gradio demo: what the model said before kick-off, against what actually happened.

Pick one of the 32 knockout ties that were really played. The app re-scores that tie with the
ratings that were LOCKED before a ball was kicked, shows the probability it gave each side, and
puts the actual result next to it.

Why this and not a free "pick any two teams" predictor: the tournament is over, so an odds
readout for Spain v England is unfalsifiable — that match never happened, and nobody can check
it. Every tie in this list has an answer, so every output can be checked.

The results table is embedded below rather than imported. A Hugging Face Space only receives the
files that are uploaded to it (app.py, match_engine.py, requirements.txt, README.md,
annex_c_third_allocation.csv and the two parquets), so anything living in 14_bracket_overlay.py
or 15_calibration.py is simply not there at runtime. The scores are a fixed historical record and
will never change again, which is what makes duplicating them safe.

Run locally:  python app.py
Deploy:       Hugging Face Space, SDK = gradio. The two parquets under data/processed/ are
              gitignored — upload them by hand or the Space crashes on load.
"""

from pathlib import Path

import base64
import os

import gradio as gr
import pandas as pd

from match_engine import match_outcome_probabilities, scoreline_grid

PROCESSED = Path(__file__).parent / "data" / "processed"
ratings = dict(pd.read_parquet(PROCESSED / "wc_final_ratings.parquet").itertuples(index=False, name=None))
params = pd.read_parquet(PROCESSED / "supremacy_params.parquet").iloc[0]
slope, total_goals = params["slope"], params["total_goals"]

# At or below this the model was not really expressing an opinion. Same threshold the carousel's
# "too close to call" slide uses, so the demo and the deck agree.
COIN_FLIP_MAX = 0.60

# (stage, team A, team B, goals A, goals B, penalties or None, who advanced)
# Ordered final-first: the top entry is the one most people will click, and the final is the
# headline — the model named both teams and the winner. Chronological order would bury it under
# thirty ties.
TIES = [
    ("Final", "Spain", "Argentina", 1, 0, None, "Spain"),
    ("Third place", "France", "England", 4, 6, None, "England"),

    ("Semi-final", "France", "Spain", 0, 2, None, "Spain"),
    ("Semi-final", "England", "Argentina", 1, 2, None, "Argentina"),

    ("Quarter-final", "France", "Morocco", 2, 0, None, "France"),
    ("Quarter-final", "Spain", "Belgium", 2, 1, None, "Spain"),
    ("Quarter-final", "Norway", "England", 1, 2, None, "England"),
    ("Quarter-final", "Argentina", "Switzerland", 3, 1, None, "Argentina"),

    ("Round of 16", "Paraguay", "France", 0, 1, None, "France"),
    ("Round of 16", "Canada", "Morocco", 0, 3, None, "Morocco"),
    ("Round of 16", "Portugal", "Spain", 0, 1, None, "Spain"),
    ("Round of 16", "United States", "Belgium", 1, 4, None, "Belgium"),
    ("Round of 16", "Brazil", "Norway", 1, 2, None, "Norway"),
    ("Round of 16", "Mexico", "England", 2, 3, None, "England"),
    ("Round of 16", "Argentina", "Egypt", 3, 2, None, "Argentina"),
    ("Round of 16", "Switzerland", "Colombia", 0, 0, (4, 3), "Switzerland"),

    ("Round of 32", "Germany", "Paraguay", 1, 1, (3, 4), "Paraguay"),
    ("Round of 32", "France", "Sweden", 3, 0, None, "France"),
    ("Round of 32", "South Africa", "Canada", 0, 1, None, "Canada"),
    ("Round of 32", "Netherlands", "Morocco", 1, 1, (2, 3), "Morocco"),
    ("Round of 32", "Portugal", "Croatia", 2, 1, None, "Portugal"),
    ("Round of 32", "Spain", "Austria", 3, 0, None, "Spain"),
    ("Round of 32", "United States", "Bosnia and Herzegovina", 2, 0, None, "United States"),
    ("Round of 32", "Belgium", "Senegal", 3, 2, None, "Belgium"),
    ("Round of 32", "Brazil", "Japan", 2, 1, None, "Brazil"),
    ("Round of 32", "Ivory Coast", "Norway", 1, 2, None, "Norway"),
    ("Round of 32", "Mexico", "Ecuador", 2, 0, None, "Mexico"),
    ("Round of 32", "England", "DR Congo", 2, 1, None, "England"),
    ("Round of 32", "Argentina", "Cape Verde", 3, 2, None, "Argentina"),
    ("Round of 32", "Australia", "Egypt", 1, 1, (2, 4), "Egypt"),
    ("Round of 32", "Switzerland", "Algeria", 2, 0, None, "Switzerland"),
    ("Round of 32", "Colombia", "Ghana", 1, 0, None, "Colombia"),
]

SHORT_STAGE = {"Round of 32": "R32", "Round of 16": "R16", "Quarter-final": "QF",
               "Semi-final": "SF", "Final": "Final", "Third place": "Third place"}

# Pre-tournament champion odds from CHAMPION_ODDS_FINAL.md — the canonical forecast, not the
# intermediate stages in CHAMPION_ODDS_ADJUSTED.md. Everyone outside the published top 20 came in
# under 0.3%, so they are reported as such rather than given a false precision.
TITLE_ODDS = {
    "Spain": 20.2, "France": 17.5, "Argentina": 13.7, "Brazil": 11.2, "England": 9.4,
    "Portugal": 7.0, "Germany": 5.9, "Belgium": 3.0, "Netherlands": 2.9, "Colombia": 1.8,
    "Morocco": 1.5, "Switzerland": 0.9, "Croatia": 0.9, "Uruguay": 0.8, "Turkey": 0.7,
    "Norway": 0.6, "Algeria": 0.3, "Austria": 0.3, "Ecuador": 0.3, "Japan": 0.3,
}
LONGSHOT = "&lt;0.3%"

FLAGS = Path(__file__).parent / "assets" / "flags"

# Same three-letter codes the bracket figure uses, so one flag set serves both.
CODE = {
    "Spain": "ESP", "France": "FRA", "Argentina": "ARG", "Brazil": "BRA", "England": "ENG",
    "Portugal": "POR", "Germany": "GER", "Belgium": "BEL", "Netherlands": "NED",
    "Colombia": "COL", "Morocco": "MAR", "Switzerland": "SUI", "Croatia": "CRO",
    "Norway": "NOR", "Paraguay": "PAR", "Sweden": "SWE", "South Africa": "RSA",
    "Canada": "CAN", "Austria": "AUT", "United States": "USA",
    "Bosnia and Herzegovina": "BIH", "Senegal": "SEN", "Japan": "JPN", "Ivory Coast": "CIV",
    "Mexico": "MEX", "Ecuador": "ECU", "DR Congo": "COD", "Cape Verde": "CPV",
    "Australia": "AUS", "Egypt": "EGY", "Algeria": "ALG", "Ghana": "GHA",
}


def flag_tag(team, height=13):
    """An <img> carrying the flag inline as a data URI.

    Base64 rather than a file path or a flag emoji, for two reasons. Gradio serves static files
    only from paths it has been told to allow, so a relative src silently 404s; and Windows
    browsers do not render regional-indicator emoji at all — 🇪🇸 shows up as the letters "ES".
    A data URI has neither problem and keeps the app self-contained on a Space.
    """
    path = FLAGS / f"{CODE.get(team, '')}.png"
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode()
    return (f'<img src="data:image/png;base64,{encoded}" height="{height}" '
            f'alt="{team}" style="vertical-align:middle;margin-right:6px">')


def label(tie):
    """One dropdown entry. The stage lives in the label so a second dropdown isn't needed."""
    stage, team_a, team_b, *_ = tie
    return f"{SHORT_STAGE[stage]}  ·  {team_a} v {team_b}"


TIE_BY_LABEL = {label(tie): tie for tie in TIES}


def expected_goals(team_a, team_b):
    """Expected goals for each side under the locked ratings. Neutral venue throughout."""
    supremacy = slope * (ratings[team_a] - ratings[team_b])
    return (max((total_goals + supremacy) / 2, 0.05),
            max((total_goals - supremacy) / 2, 0.05))


def favourite_probability(team_a, team_b):
    """Re-score one tie with the locked pre-tournament ratings.

    A knockout tie must produce a winner, so the draw probability is split evenly between the
    two sides — the same treatment the calibration diagram uses.
    """
    outcome = match_outcome_probabilities(*expected_goals(team_a, team_b))

    probability_a = outcome["home_win"] + outcome["draw"] / 2
    if probability_a >= 0.5:
        return team_a, probability_a
    return team_b, 1 - probability_a


def ordinal(n):
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


TOP_SCORELINES = 5


def scoreline_table(grid, team_a, team_b, goals_a, goals_b, correct):
    """The model's most likely scorelines, with the actual result marked.

    The rank alone says how freakish the result was; the list says what the model expected
    instead, which is the more interesting half. When the actual scoreline falls outside the top
    few it is appended after an ellipsis rather than dropped — the size of the gap IS the finding,
    and hiding it would flatter the model.
    """
    ranked = sorted(
        ((grid[i, j], i, j) for i in range(grid.shape[0]) for j in range(grid.shape[1])),
        reverse=True,
    )
    actual_rank = next(r for r, (_, i, j) in enumerate(ranked, 1) if (i, j) == (goals_a, goals_b))

    rows = ["| | Scoreline | Chance |", "|---|---|---|"]

    def row(rank, probability, i, j):
        scoreline = f"{team_a} {i}&ndash;{j} {team_b}"
        chance = format_scoreline_chance(probability)
        if (i, j) == (goals_a, goals_b):
            # The actual-result row is tinted to MATCH the verdict below — green when the favourite
            # won, red when it didn't — so the colour means one thing everywhere: green = the model
            # was right, red = it missed, exactly as on the bracket and slides. A single always-
            # green highlight read as "good" even on a miss, which is the contradiction this fixes.
            # Inline colour per cell because a markdown table gives no hook to style a whole row.
            # #4FBE8B / #E87C82 are NAVY["hit"] / NAVY["miss"] from the deck.
            tint = "color:#4FBE8B" if correct else "color:#E87C82"
            return (f'| <span style="{tint}">**{ordinal(rank)}**</span> '
                    f'| <span style="{tint}">**{scoreline}**  ← actually happened</span> '
                    f'| <span style="{tint}">**{chance}**</span> |')
        return f"| {ordinal(rank)} | {scoreline} | {chance} |"

    for rank, (probability, i, j) in enumerate(ranked[:TOP_SCORELINES], 1):
        rows.append(row(rank, probability, i, j))

    if actual_rank > TOP_SCORELINES:
        rows.append("| … | | |")
        probability, i, j = ranked[actual_rank - 1]
        rows.append(row(actual_rank, probability, i, j))

    return rows, actual_rank, grid.size


def scoreline_rank(grid, goals_a, goals_b):
    """Where the actual scoreline sat in the model's ranking of every possible scoreline.

    A bare probability is uninterpretable — 6.8% sounds low until you know that no scoreline in
    a football match ever gets much above 13%. The RANK says whether the model found this result
    ordinary or freakish, and needs no scale to read.
    """
    actual = grid[goals_a, goals_b]
    return int((grid > actual).sum()) + 1, grid.size


def format_scoreline_chance(probability):
    """Percent to one decimal, except that a genuinely rare scoreline should not print 0.0%.

    France 4-6 England really did happen and the model really did give it almost no chance —
    but "0.0%" reads as a bug rather than as a finding.
    """
    return "&lt;0.1%" if probability < 0.0005 else f"{probability * 100:.1f}%"


def title_odds(team):
    return f"{TITLE_ODDS[team]:.1f}%" if team in TITLE_ODDS else LONGSHOT


def describe_result(tie):
    """The actual score, written the way football writes it."""
    _, team_a, team_b, goals_a, goals_b, pens, advanced = tie
    if pens is None:
        return f"{team_a} {goals_a}&ndash;{goals_b} {team_b}"
    return (f"{team_a} {goals_a}&ndash;{goals_b} {team_b}, "
            f"{advanced} won {max(pens)}&ndash;{min(pens)} on penalties")


def report(selected_label):
    """Markdown for one tie: the pre-kickoff call, the result, and whether it held."""
    # Guard: the dropdown is a closed list, but a public demo should never 500 on a stray value.
    tie = TIE_BY_LABEL.get(selected_label)
    if tie is None:
        return "Pick a knockout tie from the list above."
    stage, team_a, team_b, goals_a, goals_b, pens, advanced = tie

    favourite, probability = favourite_probability(team_a, team_b)
    underdog = team_b if favourite == team_a else team_a
    correct = favourite == advanced

    expected_a, expected_b = expected_goals(team_a, team_b)
    grid = scoreline_grid(expected_a, expected_b)
    scoreline_rows, rank, total = scoreline_table(grid, team_a, team_b, goals_a, goals_b, correct)

    # "won the tie" rather than "went through": the final and the third-place match are ties too,
    # and nobody "goes through" from either of them.
    verdict = "✅ the favourite won it" if correct else "❌ the underdog won it"
    if probability <= COIN_FLIP_MAX:
        verdict += "  —  but the model called this one too close to call"

    lines = [
        # The heading is a flex row (CSS #report h3). The "v" carries its own left/right margin so
        # it sits roomy between the two tight flag+name units. The margin is inline because it is
        # about this one character, not a reusable rule — and inline HTML survives here (the flags
        # are inline HTML too).
        f'### {flag_tag(team_a)}{team_a}'
        f'<span style="margin:0 0.5rem;opacity:0.55">v</span>'
        f'{flag_tag(team_b)}{team_b}',
        f"*{stage}*",
        "",
        "**Before a ball was kicked, the model said:**",
        "",
        # Flag only in the first column — the heading directly above names both teams, so a name
        # here would just repeat it. The favourite is still the bold, top row.
        "| | Chance of winning the tie | Elo | Expected goals | Chance of winning the cup |",
        "|---|---|---|---|---|",
        f"| {flag_tag(favourite, 20)} | **{probability * 100:.0f}%** | "
        f"{ratings[favourite]:.0f} | "
        f"{(expected_a if favourite == team_a else expected_b):.2f} | {title_odds(favourite)} |",
        f"| {flag_tag(underdog, 20)} | {(1 - probability) * 100:.0f}% | "
        f"{ratings[underdog]:.0f} | "
        f"{(expected_a if underdog == team_a else expected_b):.2f} | {title_odds(underdog)} |",
        "",
        f"**What actually happened:** {describe_result(tie)}",
        "",
        f"**Scorelines it expected** — the result came {ordinal(rank)} of {total}:",
        "",
        *scoreline_rows,
        "",
        f"**{verdict}**",
    ]
    if pens is not None:
        lines += ["", "*A goals model cannot forecast a shootout either way.*"]
    return "\n".join(lines)


# Gradio's container fills the whole browser width, which on a laptop stretches three short
# lines of text across 2000px and leaves the result table stranded at the far left. Capping the
# column and centring it is the single change that makes it read as a designed page: the text
# gets a sane measure, and the table can then be told to fill that column rather than shrink to
# its own contents.
# The width is capped on an element this file owns (#page) rather than on Gradio's own container
# class, which moves between versions — 6.x still ships `.gradio-container` but the theme sets its
# own width on top. Capping a column we created works regardless of what the framework renames.
#
# The COLUMN is centred; the TEXT inside it stays left-aligned. Centred body copy gives every line
# a different starting point, so the eye has to search for the start of each one. Centring the
# block and left-aligning its contents is the combination that reads as designed.
CSS = """
#page { max-width: 720px; margin: 0 auto; }
#report table { width: 100%; border-collapse: collapse; margin: 0.5rem 0 1rem; }
#report th, #report td { padding: 0.5rem 0.75rem; }
#report th:last-child, #report td:last-child { text-align: right; }
/* The heading is a flex row, not flowing text. white-space:nowrap did not hold — Gradio's prose
   CSS wins that battle — so the matchup kept breaking after the "v". A flex row with flex-wrap:
   nowrap keeps flags and names on one line no matter what the framework sets on white-space. */
/* gap:0 — spacing is set per element instead, because a flag and its country name want to sit
   TIGHT (they are one unit) while the "v" wants room on both sides. A uniform flex gap made all
   four spaces equal, which floated each flag away from its name. */
#report h3 { display: flex; flex-wrap: nowrap; align-items: center; gap: 0;
             margin: 0 0 0.15rem; }
/* Size the flags by CSS, not the height attribute: Gradio strips height from a heading img (it
   keeps it inside a table), which left them at full size. A CSS rule cannot be stripped. The
   small right margin is the flag-to-name gap. */
#report h3 img { height: 20px; width: auto; margin-right: 0.3rem; }
/* The stage line is the heading's subtitle. As plain italics it read as an afterthought; a small
   upper-case label in the accent colour makes it a caption without competing with the matchup. */
#report h3 + p em { font-style: normal; text-transform: uppercase; letter-spacing: 0.06em;
                    font-size: 0.72rem; font-weight: 600; opacity: 0.75; }
#report p { margin: 0.45rem 0; }
#footer { font-size: 0.9rem; opacity: 0.75; margin-top: 0.5rem; }
"""

# The headline is computed, not typed. It is the same claim the carousel makes, and a number that
# drifts from the code is exactly the failure the deck's slide 4 already had once.
FAVOURITE_WINS = sum(1 for tie in TIES if favourite_probability(tie[1], tie[2])[0] == tie[6])

with gr.Blocks(title="World Cup 2026 — what the model said", fill_width=False) as demo:
    with gr.Column(elem_id="page"):
        gr.Markdown(
            "# World Cup 2026: what my model said before kick-off\n"
            "Elo-driven Dixon&ndash;Coles Poisson model, **locked before the tournament started** "
            "(the git history is the timestamp). Across all "
            f"{len(TIES)} knockout ties the favourite it named won **{FAVOURITE_WINS}**.\n\n"
            "Pick any tie below and compare the forecast with what actually happened."
        )
        # filterable=False makes it a pure click-to-select — no text box to type in. The default
        # dropdown is a type-to-search field, which looks editable even though allow_custom_value is
        # off; for a fixed list of 32 this is cleaner and matches what a "pick one" control should be.
        tie_choice = gr.Dropdown(
            list(TIE_BY_LABEL),
            value=label(TIES[0]),
            label="Knockout tie",
            filterable=False,
            allow_custom_value=False,
        )
        output = gr.Markdown(elem_id="report")
        gr.Markdown(
            "---\n"
            "Code, data and the full report card: "
            "[github.com/Ganapathy-K/fifa-world-cup-2026-forecast]"
            "(https://github.com/Ganapathy-K/fifa-world-cup-2026-forecast)",
            elem_id="footer",
        )

    # Bound to .change() rather than to a button: the answer updates the instant the selection
    # does, so there is never a moment where an old result sits under a new choice. The earlier
    # version of this app had exactly that bug and a stale panel reads as a wrong answer.
    tie_choice.change(report, tie_choice, output)
    demo.load(report, tie_choice, output)

if __name__ == "__main__":
    # Gradio 6 moved css from the Blocks constructor to launch(). Passing it to Blocks is not an
    # error — it warns and then silently does nothing, which is why the page stayed full-width
    # while the stylesheet was visibly being served.
    #
    # Cloud Run injects PORT and routes to the container's external interface, so the server has
    # to bind 0.0.0.0 rather than Gradio's default 127.0.0.1 — bound to localhost the container
    # starts cleanly and then fails every health check. Locally PORT is unset and 7860 applies.
    demo.launch(css=CSS, server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
