"""
One-off: scrape FIFA's official Annex C third-place allocation table (all 495
combinations) from the Wikipedia knockout-stage page and save it as a clean local
reference CSV. This is what makes the bracket FIFA-exact and reproducible instead of
an in-house approximation.

Output: annex_c_third_allocation.csv with columns
    combo (8 sorted qualifying group letters, e.g. "BCEFGHIJ"), A, B, D, E, G, I, K, L
where each winner column holds the third-placed GROUP letter that winner faces.

Run once: .venv/Scripts/python.exe fifa_wc_2026_poisson/build_annex_c_table.py
"""

import io
import urllib.request
import pandas as pd
from pathlib import Path

URL = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_knockout_stage"
OUTPUT_PATH = Path(__file__).parent / "annex_c_third_allocation.csv"
WINNER_COLUMNS = ["1A vs", "1B vs", "1D vs", "1E vs", "1G vs", "1I vs", "1K vs", "1L vs"]

request = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
html = urllib.request.urlopen(request).read().decode("utf-8")
table = pd.read_html(io.StringIO(html))[0]
assert table.shape[0] == 495, f"expected 495 combinations, got {table.shape[0]}"

group_columns = [c for c in table.columns if "advance from groups" in str(c)]
assert len(group_columns) == 12, f"expected 12 group columns, got {len(group_columns)}"

records = []
for _, row in table.iterrows():
    qualifying_groups = sorted(str(row[c]).strip() for c in group_columns if pd.notna(row[c]))
    assert len(qualifying_groups) == 8, f"row has {len(qualifying_groups)} qualifiers"
    record = {"combo": "".join(qualifying_groups)}
    for winner_column in WINNER_COLUMNS:
        winner_group = winner_column[1]                 # "1A vs" -> "A"
        third_group = str(row[winner_column]).strip()[-1]   # "3H" -> "H"
        record[winner_group] = third_group
    records.append(record)

annex_c = pd.DataFrame(records)
assert annex_c["combo"].is_unique, "combinations should be unique"
annex_c.to_csv(OUTPUT_PATH, index=False)

# Validate against the Row-109 mapping fetched manually (combo BCEFGHIJ)
check = annex_c[annex_c["combo"] == "BCEFGHIJ"].iloc[0]
expected = {"A": "H", "B": "G", "D": "B", "E": "C", "G": "J", "I": "F", "K": "E", "L": "I"}
actual = {w: check[w] for w in expected}
assert actual == expected, f"Row-109 mismatch: {actual}"

print(f"Saved {len(annex_c)} combinations -> {OUTPUT_PATH.name}")
print(f"Validation BCEFGHIJ -> {actual}  (matches FIFA Annex C Row 109)")
