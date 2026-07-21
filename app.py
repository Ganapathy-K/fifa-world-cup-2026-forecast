"""Gradio demo for the FIFA World Cup 2026 Poisson predictor.

Wraps the locked match engine (match_engine.py + the committed parquet ratings)
in a two-dropdown UI: pick two teams and get win/draw/loss odds plus the most
likely scorelines. Host teams (USA/Mexico/Canada) get the home-advantage bump
when listed as the first (home) team, exactly as in _tooling/predict_match.py.

Run locally:  python app.py
Deploy:       push this file + requirements.txt to a Hugging Face Space (SDK: gradio).
"""

from pathlib import Path

import gradio as gr
import pandas as pd

from match_engine import match_outcome_probabilities, scoreline_grid

PROCESSED = Path(__file__).parent / "data" / "processed"
ratings = dict(pd.read_parquet(PROCESSED / "wc_final_ratings.parquet").itertuples(index=False, name=None))
params = pd.read_parquet(PROCESSED / "supremacy_params.parquet").iloc[0]
slope, intercept, total_goals = params["slope"], params["intercept"], params["total_goals"]

HOSTS = {"United States", "Mexico", "Canada"}
TEAMS = sorted(ratings)


def expected_goals(home_team, away_team, neutral=False):
    """Expected goals for each side, giving the home team the host bump when applicable."""
    rating_gap = ratings[home_team] - ratings[away_team]
    home_advantage = 0.0 if (neutral or home_team not in HOSTS) else intercept
    supremacy = slope * rating_gap + home_advantage
    return (
        max((total_goals + supremacy) / 2, 0.05),
        max((total_goals - supremacy) / 2, 0.05),
    )


def predict(home_team, away_team, neutral_venue):
    """Return a markdown report of match odds and the most likely scorelines."""
    if home_team == away_team:
        return "### Pick two different teams."

    expected_home, expected_away = expected_goals(home_team, away_team, neutral_venue)
    outcome = match_outcome_probabilities(expected_home, expected_away)
    host_note = "" if (neutral_venue or home_team not in HOSTS) else " *(+host advantage)*"

    grid = scoreline_grid(expected_home, expected_away)
    scorelines = sorted(
        ((grid[i, j], i, j) for i in range(grid.shape[0]) for j in range(grid.shape[1])),
        reverse=True,
    )[:6]

    lines = [
        f"### {home_team} vs {away_team}{host_note}",
        f"**Elo:** {home_team} {ratings[home_team]:.0f} &nbsp;|&nbsp; {away_team} {ratings[away_team]:.0f}",
        f"**Expected goals:** {home_team} {expected_home:.2f} &ndash; {expected_away:.2f} {away_team}",
        "",
        "| Outcome | Probability |",
        "|---|---|",
        f"| **{home_team} win** | {outcome['home_win'] * 100:.1f}% |",
        f"| Draw | {outcome['draw'] * 100:.1f}% |",
        f"| **{away_team} win** | {outcome['away_win'] * 100:.1f}% |",
        "",
        "**Most likely scorelines**",
        "",
        "| Scoreline | Probability |",
        "|---|---|",
    ]
    lines += [f"| {home_team} {i}&ndash;{j} {away_team} | {p * 100:.1f}% |" for p, i, j in scorelines]
    return "\n".join(lines)


with gr.Blocks(title="FIFA World Cup 2026 Predictor") as demo:
    gr.Markdown(
        "# FIFA World Cup 2026 Match Predictor\n"
        "Elo-driven Dixon&ndash;Coles Poisson model. The **first team is the home side** &mdash; "
        "hosts (USA, Mexico, Canada) get a home-advantage bump unless you tick *neutral venue*."
    )
    with gr.Row():
        home = gr.Dropdown(TEAMS, value="Mexico", label="Home team")
        away = gr.Dropdown(TEAMS, value="South Africa", label="Away team")
    neutral = gr.Checkbox(value=False, label="Neutral venue (no host advantage)")
    go = gr.Button("Predict", variant="primary")
    report = gr.Markdown()

    go.click(predict, [home, away, neutral], report)
    demo.load(predict, [home, away, neutral], report)

if __name__ == "__main__":
    demo.launch()
