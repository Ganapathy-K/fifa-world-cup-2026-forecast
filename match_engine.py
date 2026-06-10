"""
Shared match engine for the FIFA WC 2026 predictor.

Single source of truth for turning two teams' expected goals into (a) a scoreline
probability grid, (b) a knockout win probability, and (c) sampled scorelines for the
Monte Carlo. Every notebook/script (03, 05, 08, 09, 10, 11) imports from here so the
goal model lives in exactly one place.

Goal model: independent Poisson with the Dixon-Coles low-score correction.
Plain Poisson treats the two teams' goals as independent, which empirically produces
too few low-scoring draws (0-0, 1-1) and too many 1-0 / 0-1 results. Dixon & Coles
(1997) multiply the four low-score cells by a correction term tau() governed by a
single parameter rho:

    tau(0, 0) = 1 - lambda * mu * rho
    tau(0, 1) = 1 + lambda * rho
    tau(1, 0) = 1 + mu * rho
    tau(1, 1) = 1 - rho
    tau(x, y) = 1                       otherwise

where lambda = expected home goals, mu = expected away goals. With rho < 0 this lifts
0-0 and 1-1 and trims 1-0 / 0-1, matching real football. After applying tau the grid
no longer sums to 1, so we renormalise.
"""

import csv
from pathlib import Path

import numpy as np
from scipy.stats import poisson

MAX_GOALS = 10            # scoreline grid runs 0..MAX_GOALS for each side
DIXON_COLES_RHO = -0.13   # Dixon & Coles (1997) estimate; rho < 0 boosts low-score draws


def _dixon_coles_correction(grid, expected_home_goals, expected_away_goals, rho):
    """Multiply the four low-score cells of a grid by the Dixon-Coles tau factors.

    Works for a single match (2-D grid, scalar expected goals) and for a batch of
    fixtures (3-D grid shaped (n_fixtures, G, G) with expected-goal arrays), because
    the corner indexing broadcasts the same way in both cases.
    """
    grid[..., 0, 0] *= 1.0 - expected_home_goals * expected_away_goals * rho
    grid[..., 0, 1] *= 1.0 + expected_home_goals * rho
    grid[..., 1, 0] *= 1.0 + expected_away_goals * rho
    grid[..., 1, 1] *= 1.0 - rho
    return grid


def scoreline_grid(expected_home_goals, expected_away_goals,
                   max_goals=MAX_GOALS, rho=DIXON_COLES_RHO):
    """Dixon-Coles corrected joint scoreline distribution for one match.

    grid[i, j] = P(home scores i AND away scores j). Normalised to sum to 1.
    """
    goals = np.arange(0, max_goals + 1)
    home_goal_probs = poisson.pmf(goals, expected_home_goals)
    away_goal_probs = poisson.pmf(goals, expected_away_goals)
    grid = np.outer(home_goal_probs, away_goal_probs)
    if rho != 0.0:
        grid = _dixon_coles_correction(grid, expected_home_goals, expected_away_goals, rho)
    return grid / grid.sum()


def win_probability(expected_home_goals, expected_away_goals,
                    max_goals=MAX_GOALS, rho=DIXON_COLES_RHO):
    """P(home wins | not a draw) — the knockout resolver. Draw mass is removed."""
    grid = scoreline_grid(expected_home_goals, expected_away_goals, max_goals, rho)
    home_wins = np.tril(grid, -1).sum()   # home goals > away goals
    away_wins = np.triu(grid, 1).sum()    # away goals > home goals
    return home_wins / (home_wins + away_wins)


def match_outcome_probabilities(expected_home_goals, expected_away_goals,
                                max_goals=MAX_GOALS, rho=DIXON_COLES_RHO):
    """Collapse the grid into (home_win, draw, away_win) probabilities."""
    grid = scoreline_grid(expected_home_goals, expected_away_goals, max_goals, rho)
    return {
        "home_win": np.tril(grid, -1).sum(),
        "draw": np.trace(grid),
        "away_win": np.triu(grid, 1).sum(),
    }


def build_scoreline_sampler(expected_home_goals, expected_away_goals,
                            max_goals=MAX_GOALS, rho=DIXON_COLES_RHO):
    """Precompute a fast inverse-CDF sampler for a fixed set of fixtures.

    Expected goals never change across Monte Carlo runs, so we build each fixture's
    Dixon-Coles scoreline distribution once and reuse it. Returns a tuple consumed by
    sample_scorelines(): (cumulative_probs, home_goal_lookup, away_goal_lookup).
    """
    expected_home_goals = np.asarray(expected_home_goals, dtype=float)
    expected_away_goals = np.asarray(expected_away_goals, dtype=float)
    goals = np.arange(0, max_goals + 1)
    home_goal_probs = poisson.pmf(goals[None, :], expected_home_goals[:, None])
    away_goal_probs = poisson.pmf(goals[None, :], expected_away_goals[:, None])
    grid = home_goal_probs[:, :, None] * away_goal_probs[:, None, :]   # (n, G, G)
    if rho != 0.0:
        grid = _dixon_coles_correction(grid, expected_home_goals, expected_away_goals, rho)
    flat = grid.reshape(grid.shape[0], -1)
    flat /= flat.sum(axis=1, keepdims=True)
    cumulative_probs = np.cumsum(flat, axis=1)
    home_goal_lookup = np.repeat(goals, max_goals + 1)   # flat index -> home goals
    away_goal_lookup = np.tile(goals, max_goals + 1)     # flat index -> away goals
    return cumulative_probs, home_goal_lookup, away_goal_lookup


def sample_scorelines(sampler, rng):
    """Draw one scoreline per fixture from a prebuilt sampler. Vectorised."""
    cumulative_probs, home_goal_lookup, away_goal_lookup = sampler
    draws = rng.random(cumulative_probs.shape[0])
    flat_index = (draws[:, None] > cumulative_probs).sum(axis=1)
    flat_index = np.clip(flat_index, 0, cumulative_probs.shape[1] - 1)
    return home_goal_lookup[flat_index], away_goal_lookup[flat_index]


def sample_scoreline(expected_home_goals, expected_away_goals, rng,
                     max_goals=MAX_GOALS, rho=DIXON_COLES_RHO):
    """Draw a single (home_goals, away_goals) scoreline for one match."""
    sampler = build_scoreline_sampler([expected_home_goals], [expected_away_goals],
                                      max_goals, rho)
    home_goals, away_goals = sample_scorelines(sampler, rng)
    return int(home_goals[0]), int(away_goals[0])


# ----------------------------------------------------------------------------
# Round of 32 bracket — official FIFA WC 2026 structure
# ----------------------------------------------------------------------------
# 48 teams: 12 group winners + 12 runners-up + 8 best third-placed teams.
#
# THIRD_PLACE_SLOTS: the eight group winners that host a third-placed team, each
# keyed to the set of third-placed GROUPS it is allowed to draw (FIFA's published
# lists). The winner's own group is never in its set, so a same-group meeting is
# impossible. Which of the 8 qualifying thirds lands in which slot is FIFA's Annex C
# (495 combinations); we reproduce a *valid* assignment via bipartite matching, which
# is identical to Annex C for aggregate probabilities even when several are legal.
THIRD_PLACE_SLOTS = {
    "E": set("ABCDF"),   # match 74
    "I": set("CDFGH"),   # match 77
    "A": set("CEFHI"),   # match 79
    "L": set("EHIJK"),   # match 80
    "D": set("BEFIJ"),   # match 81
    "G": set("AEHIJ"),   # match 82
    "B": set("EFGIJ"),   # match 85
    "K": set("DEIJL"),   # match 87
}

# The 16 Round-of-32 matches in *bracket order* — consecutive pairs feed the same
# Round-of-16 match, those pairs feed the same quarter-final, and so on, so a plain
# sequential single-elimination loop reproduces the official knockout tree to the
# final. Each slot is ("W", group) winner, ("R", group) runner-up, or ("T", group)
# the third-placed team hosted by that winner.
RO32_BRACKET = [
    (("W", "E"), ("T", "E")),   # 74: 1E v 3rd
    (("W", "I"), ("T", "I")),   # 77: 1I v 3rd
    (("R", "A"), ("R", "B")),   # 73: 2A v 2B
    (("W", "F"), ("R", "C")),   # 75: 1F v 2C
    (("R", "K"), ("R", "L")),   # 83: 2K v 2L
    (("W", "H"), ("R", "J")),   # 84: 1H v 2J
    (("W", "D"), ("T", "D")),   # 81: 1D v 3rd
    (("W", "G"), ("T", "G")),   # 82: 1G v 3rd
    (("W", "C"), ("R", "F")),   # 76: 1C v 2F
    (("R", "E"), ("R", "I")),   # 78: 2E v 2I
    (("W", "A"), ("T", "A")),   # 79: 1A v 3rd
    (("W", "L"), ("T", "L")),   # 80: 1L v 3rd
    (("W", "J"), ("R", "H")),   # 86: 1J v 2H
    (("R", "D"), ("R", "G")),   # 88: 2D v 2G
    (("W", "B"), ("T", "B")),   # 85: 1B v 3rd
    (("W", "K"), ("T", "K")),   # 87: 1K v 3rd
]


def _load_annex_c():
    """Load FIFA's official Annex C third-place allocation (495 combinations) from the
    local reference CSV (built by build_annex_c_table.py). Returns
    {frozenset(qualifying group letters): {winner_group: third_group}} or None."""
    path = Path(__file__).parent / "annex_c_third_allocation.csv"
    if not path.exists():
        return None
    table = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            table[frozenset(row["combo"])] = {w: row[w] for w in ("A", "B", "D", "E", "G", "I", "K", "L")}
    return table


ANNEX_C = _load_annex_c()


def assign_thirds_to_winners(qualified_third_groups):
    """Map the 8 qualifying third-placed GROUPS to the 8 winner slots that host them.
    Uses FIFA's official Annex C table (exact) when available; otherwise falls back to a
    valid bipartite matching honouring THIRD_PLACE_SLOTS. Returns {winner: third}."""
    qualified = frozenset(qualified_third_groups)
    if ANNEX_C is not None and qualified in ANNEX_C:
        return dict(ANNEX_C[qualified])
    assignment = {}
    used = set()

    def options(slot):
        # sorted() makes the assignment deterministic: iterating a Python set of
        # strings has process-dependent order (hash randomisation), which would
        # otherwise pick a different valid bracket each run and shift the odds.
        return sorted(g for g in THIRD_PLACE_SLOTS[slot] if g in qualified and g not in used)

    def backtrack(remaining_slots):
        if not remaining_slots:
            return True
        slot = min(remaining_slots, key=lambda s: len(options(s)))
        for third_group in options(slot):
            assignment[slot] = third_group
            used.add(third_group)
            if backtrack([s for s in remaining_slots if s != slot]):
                return True
            used.discard(third_group)
            del assignment[slot]
        return False

    if not backtrack(list(THIRD_PLACE_SLOTS.keys())):
        raise ValueError(f"No valid third-place assignment for groups {sorted(qualified)}")
    return assignment


def build_round_of_32(winners, runners, thirds_by_group):
    """Build the official Round-of-32 matches in bracket order.

    winners, runners : {group_letter: team_name} for all 12 groups.
    thirds_by_group  : {group_letter: team_name} for the 8 qualifying third groups.
    Returns a list of 16 (team_a, team_b) tuples ready for sequential elimination.
    """
    third_assignment = assign_thirds_to_winners(thirds_by_group.keys())

    def resolve(slot):
        kind, group = slot
        if kind == "W":
            return winners[group]
        if kind == "R":
            return runners[group]
        return thirds_by_group[third_assignment[group]]   # "T"

    return [(resolve(slot_a), resolve(slot_b)) for slot_a, slot_b in RO32_BRACKET]
