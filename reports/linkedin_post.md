# LinkedIn post — WC 2026 report card

Companion text for the 5-slide carousel (`reports/figures/wc2026_report_card_carousel.pdf`).
Plain text on purpose: LinkedIn strips bold/italics, so the shape is carried by line breaks and
blank lines only. Paste, check the preview, and if LinkedIn eats a blank line, retype it in the box.

Best slot: Tue–Thu, ~11am–12:30pm IST. Post the carousel; drop the Hugging Face demo link as a
comment once the Space is live.

---

My World Cup forecast called the champion and the exact final — before a ball was kicked.

Here's the report card, misses included.

It got the final right (Spain over Argentina) and 26 of 32 knockout ties.

It also had Brazil, Germany and Portugal going further than they did. Only two were genuine upsets, though — Germany and Brazil both lost as clear favourites. Three exits came down to penalty shootouts, which a goals model can't call either way.

The honest bit: it was under-confident. The five matches it was surest about all happened; the ten it called coin-flips went 6–4.

What I'd change next time: give penalties a coin-flip instead of a blank, and stop being so timid with the probabilities.

Method: Elo-driven Dixon–Coles Poisson, 10,000 sims, locked in git before kickoff — the history is the timestamp.

Code + data: github.com/Ganapathy-K/fifa-world-cup-2026-forecast
