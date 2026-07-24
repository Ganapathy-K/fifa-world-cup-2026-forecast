# LinkedIn post — FIFA WC 2026 report card

**POSTED 2026-07-24, ~20:00 IST.** This is the text exactly as it went out, kept as the record.

Plain text on purpose: LinkedIn strips bold and italics, so the shape is carried by line breaks
alone. It also eats blank lines on paste — every one below had to be retyped in the box.

Attached as a **document post** (not images): `reports/figures/wc2026_report_card_carousel.pdf`,
5 pages, 4:5. Document title given at upload: *FIFA World Cup 2026 — the report card*.

The demo link was deliberately NOT in the body — it went in a comment once Cloud Run was live,
along with a link back to the 11 June post that made the original call.

---

My FIFA World Cup 2026 forecast called the champion and the exact final — before a ball was kicked.

Here's the report card, misses included.

It got the final right (Spain over Argentina) and 26 of 32 knockout ties.

It also had Brazil, Germany and Portugal going further than they did. Only two were genuine upsets, though — Germany and Brazil both lost as clear favourites. Three exits came down to penalty shootouts, which a goals model can't call either way.

The honest bit: it was underconfident. The five matches it was surest about all happened. The ten it called coin-flips went 6–4.

What I'd change next time: give penalties a coin-flip instead of a blank, and stop being so timid with the probabilities.

Method: Elo-driven Dixon–Coles Poisson, 10,000 sims, locked in git before kickoff — the history is the timestamp.

Code + data: https://lnkd.in/gsZdfYAv

#MachineLearning #DataScience #FIFAWorldCup #Forecasting #AI

---

`lnkd.in/gsZdfYAv` is LinkedIn's own shortener — it rewrote the plain
`github.com/Ganapathy-K/fifa-world-cup-2026-forecast` on submit. Nothing was pasted shortened.

## Edited in place ~22:40, about 2.5 h after publishing
Two reactions in the first 2.5 h prompted it. Claude's advice was to leave it alone — distribution
is allocated in the first hour or two and an edit does not re-run it, so late hashtags reach almost
nobody. **He chose to edit anyway rather than spend a separate post on it, which is a reasonable
call: the cost is near zero and the post is the artifact he will link from applications for months.**
Two changes: `My World Cup forecast` → **`My FIFA World Cup 2026 forecast`** (the tournament was
never named, and "World Cup" is ambiguous for someone who also posts cricket), and five hashtags
added at the very bottom so they cannot break the hook. Five is the practical ceiling.

## Follow-up comment (posted under the 11 June original, replying to the demo request)

> Hey! Here's the demo where you could pick any knockout match (from R32 right up to the final)
> and compare pre-kickoff forecast with what actually happened:
> https://fifa-wc-2026-report-card-445269150468.us-central1.run.app
>
> Follow up FWC 2026 post — report card here: [link to the 24 July post]

## Edits made against the earlier draft
- `under-confident` → `underconfident` (one word, and it matches slide 5).
- The semicolon in that line became a full stop — a shorter sentence reads faster in a feed.
