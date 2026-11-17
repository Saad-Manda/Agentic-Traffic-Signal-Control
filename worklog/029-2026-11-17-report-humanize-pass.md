# 029 — Report humanize pass (targeted)

**Date:** 2026-11-17 · **Phase:** 9.1 · **Type:** report

## What
A targeted editing pass over `report/chapters/*.tex` following an outside critique that rated
the report as likely AI-edited (~65/100). Fixed the specific patterns it named, then ran the
humanizer skill's checks after each chapter's rewrite.

## Why
The critique pointed at real writing faults: five or more statements about the report's own
honesty, the Com-MP / Hoel et al. comparison made three times (§1.5, §2.1.4, §2.2), summary lines
at the ends of paragraphs, and short caveat sentences added after a claim. User decisions (17
Nov): keep the impersonal voice, add no first-hand anecdotes, change the report only (not the
poster), and keep the edit targeted rather than a full rewrite.

## How
- Removed the honesty statements: "negative results are reported as they came out" (§1.4),
  "H3 is not, and the report keeps that result" (§1.5), "it is a loss and is reported as one"
  (§4.2), "each of its lines was checked against the code" (§3.5), "this claim holds only as far
  as that search reached" (§2.2, merged into gap 2 as "was not exhaustive, but"). Dropped the
  repeated "run once on locked models" from §1.5 and the Ch 4 opening (kept in §3.2 and the abstract).
- The Com-MP / Hoel comparison now appears once, in the last paragraph of §2.2 (new label
  `sec:gaps`). §1.5 item 3 points to it, §2.1.4 only describes Com-MP, and gap 2 no longer repeats it.
- Cut or rewrote summary lines at paragraph ends: "In short, σ responds…" (§4.4), "wins none… never far
  behind" (§4.3, now one sentence with the regrets), the §1.1 closing triad, the §2.1.2 closer.
- Merged short caveat sentences into the sentence before them: §3.4.3 (attention is an association,
  now points to the faithfulness test), §3.4.4 ("It is a rough estimate." list turned into prose),
  §3.4.6 safety caveats, §4.5.3 and §4.5.4 short sentences, §5.2 ethics, §5.3 items (3) and (4) as full
  sentences. §5.1 "What the uncertainty measures" without the repeated openings.
- Abstract: left unchanged; no flagged pattern.

## Result
`latexmk` builds cleanly, **13 pages**, `scripts/check_report.py` ok. No number, `\res` macro,
citation or claim changed; `report/checks/claims.md` needs no update (no typed number moved
except "12 or 16 intersections", which is a fact from Table 3.1).

## Files changed
| File | Action | What changed | Why |
|---|---|---|---|
| `report/chapters/01-introduction.tex` | modified | §1.1 closing sentence; §1.2 last sentence; §1.4 honesty clause cut; §1.5 item 3 comparison replaced by pointer to §2.2; item 5 honesty clause and "run once" cut | repetition, honesty statements, closers |
| `report/chapters/02-literature.tex` | modified | §2.1.2 closer cut; §2.1.4 Com-MP sentence shortened; §2.2 label `sec:gaps`, gaps 1–2 shortened, search-scope caveat merged; final paragraph holds the only Com-MP / Hoel comparison | single comparison, fewer caveats |
| `report/chapters/03-methodology.tex` | modified | attention-weight caveat (§3.4.3), σ limits (§3.4.4), safety caveats (§3.4.6) as prose; "checked against the code" cut (§3.5) | caveat lists, honesty statement |
| `report/chapters/04-results.tex` | modified | chapter opening; §4.2 "reported as one" cut; §4.3 regret sentence; §4.4 "In short" cut; error analysis and explainability sentences merged | closers, honesty statements |
| `report/chapters/05-conclusion.tex` | modified | §5.1 σ bullet; §5.2 ethics sentence merged; §5.3 items (3)–(4) as sentences | repeated openings, fragments |
| `report/report.pdf` | modified | rebuilt | reflects the edits |
| `worklog/029-…`, `worklog/README.md` | created / modified | this entry and its index row | work log rule |

## Commits
Not committed yet.

## Next
User reads the changed passages and rewords anything that does not sound like their own voice.
