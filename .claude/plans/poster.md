# Plan — Poster (ROADMAP 9.2, 15 Nov 2026)

Source: user request 15 Nov. Decisions taken with the user the same day:

| Question | Decision |
|---|---|
| Tool | PowerPoint file built by a script (`python-pptx`); no Canva / Figma |
| Submission | `poster/poster.pptx` (editable) + `poster/poster.pdf` |
| Display | on screen / projector → one 16:9 slide, 13.33 × 7.5 in (same as the sample) |
| Design | fully free; `updated poster sample.pptx` is a sample only, not a template |
| Logos | none (user, 15 Nov) — no IIT Patna / IMPACT logo on the poster |

Serves rubric rows "poster" / "results and discussion" and guideline G12 (communication).
ROADMAP 9.2 still says "on the template, keep its header band, colours and box layout" — that
text is replaced and a Decision Log row is added (step 8).

## Goal

One slide that a viewer can read in about two minutes and that covers the whole project:
problem, limits of the conventional frameworks, the method, the protocol, the test results
with the negative ones, and the conclusion. Numbers come from `docs/results/*.csv`, never typed.

## Justification

- **Script, not hand layout.** Project rule: every table and figure is produced by a command in
  the README. A script also keeps poster numbers identical to the report (same CSVs).
- **`python-pptx`, not `pptxgenjs`.** The project is Python; one more pip package instead of a
  Node toolchain. Neither is installed today.
- **Canva rejected:** prompt-driven generation, no exact placement of tables / equations / own
  figures, not reproducible. **Figma rejected:** no connector, hand layout.
- **LaTeX poster rejected:** would reuse the `\res` macros directly, but gives no editable PPTX,
  which is the format the sample was issued in.

## Design

**Canvas:** 13.33 × 7.5 in, margins 0.25 in. Fonts from the safe list (Cambria headings,
Calibri body) so the LibreOffice render matches PowerPoint. Body text ≥ 9 pt, table / caption
text ≥ 8 pt, section headings 13–14 pt, title 22–24 pt (the sample uses about the same sizes;
the slide is viewed full-screen).

**Colour:** dark slate header, white body, one accent for UG-MARL in every table row, plot and
diagram; baselines in greys / muted tones. Same model → same colour everywhere
(`trafficrl.utils` plotting style is the source for model colours). No decorative bars.

**Layout** (results get the most area; widths approximate):

```
+--------------------------------------------------------------------------------------+
|        Title · name · Roll No. · e-mail · department, institute (text only)          |
+----------------------+-------------------------------+-------------------------------+
| 1 Problem            | 3 Method: UG-MARL             | 5 Results (Jinan test, 5 seeds)|
|   2-3 lines + PS16   |   architecture figure         |   headline numbers (3 tiles)  |
| 2 Limits of existing |   gate rule (equation)        |   Table 1: main comparison    |
|   frameworks (4 rows:|   Algorithm 1 (compact, 6-7   |   Fig 2: robustness / regret  |
|   FT, MP, DQN, IDQN) |   lines)                      |   Fig 3: ablation             |
|   Hypotheses H1-H3   | 4 Setup strip: data, splits,  | 6 Held-out city + gate firing |
|   Contributions (3)  |   baselines, protocol         |   (one small plot + 2 lines)  |
+----------------------+-------------------------------+-------------------------------+
| 7 Findings: H1 yes · H2 yes · H3 no | Limitations | Conclusion | References (4-5)    |
+--------------------------------------------------------------------------------------+
```

**Content per block** (text is taken from the report chapters and shortened; impersonal voice,
no "we" / "I"):

1. Problem — RL signal control has no reliability signal and degrades under input shift;
   Max-Pressure is stable but does not learn. Source: `report/chapters/01-introduction.tex`.
2. Limits of existing frameworks — one line each for Fixed-Time, Max-Pressure, shared DQN,
   IDQN; H1–H3 in one line each; the three contributions from §1.6.
3. Method — `report/figures/architecture.png`; gate rule
   a = argmax mean-Q if σ(s) ≤ τ else Max-Pressure, with σ = std over K heads and τ = p-th
   percentile of σ on validation; compact Algorithm 1 from `report/figures/algorithm.tex`.
4. Setup — Jinan 3×4 (train / val / test), Hangzhou 4×4 held out; CityFlow; Δt = 15 s; reward
   −|pressure|; seeds 0–4; selection on validation only, test and held-out run once.
5. Results — three headline tiles (clean test travel time vs Max-Pressure and IDQN; travel time
   at noise 0.2; worst-case regret 6.8 s). Table 1 from `test_main.csv` (5 models: travel time,
   waiting time, max wait, params; mean ± std). Fig 2 from `test_robustness.csv`, Fig 3 from
   `test_ablation.csv`.
6. Generalisation and explainability — Hangzhou result from `heldout_generalisation.csv`; gate
   firing rate by condition from `test_safety_explain.csv`.
7. Findings / limitations / conclusion — H1 supported; H2 supported (clean max-wait part inside
   seed noise); H3 not supported (−attention 294.2 vs 295.4 s, 63 % fewer parameters); 3.2 s
   behind DQN on clean test, not caused by the gate; gate raises max wait under the surge
   (2162 vs 1430 s); ungated policies leave 11–14 % of vehicles outside the network at noise
   0.2 and wait metrics count in-network waiting only. References: 4–5 of the report's 10.

**Figures.** The report figures are sized for an A4 column. Poster versions are re-rendered
with larger fonts by calling the existing `trafficrl.eval.figures` functions (`fig_robustness`
or `fig_regret`, `fig_ablation`, `fig_gate_firing`) under a poster `rc_context`, saved as PNG
at 300 dpi into `poster/figures/`. They read the saved logs in `results/final/`; nothing is
simulated. Which of robustness / regret is used is decided on the first render (the one that
stays legible at ~4 in wide).

**Table 1** is a native PowerPoint table (editable), UG-MARL row tinted, best value per column
bold. Cell strings come from the same formatting helpers as the report tables
(`trafficrl.eval.latex` / `report.py`), without the LaTeX markup.

**Logos.** None. The header is text only (institute and department named in the author
line), so nothing is taken from `updated poster sample.pptx`.

## Files

| File | Action | Purpose |
|---|---|---|
| `src/trafficrl/eval/poster.py` | create | pure helpers: load CSV rows → table cells and headline numbers; poster figure rendering (calls `figures.py`) |
| `scripts/build_poster.py` | create | lays out the slide with `python-pptx`, writes `poster/poster.pptx`; `--pdf` converts with LibreOffice |
| `poster/content.yaml` | create | all poster prose (title, bullets, captions, references) so text edits need no code change |
| `poster/figures/*.png` | generated | poster-size plots |
| `poster/poster.pptx`, `poster/poster.pdf` | generated, tracked | deliverables |
| `tests/eval/test_poster.py` | create | see Tests |
| `requirements.txt`, `pyproject.toml` | modify | add pinned `python-pptx` |
| `README.md` | modify | poster build command (§ report / poster) |
| `ROADMAP.md` | modify | 9.2 text + tick, Decision Log row (template → free design, script-built) |
| `CLAUDE.md` | modify | Phase 9 commands, `poster/` line in the structure, note on the poster |
| `report/checks/claims.md` | modify | add poster-typed numbers, if any are typed in `content.yaml` |
| `worklog/026-2026-11-15-poster.md`, `worklog/README.md` | create / modify | work log |

## Config

No YAML experiment config and no seed: the build is a pure reader. `poster/content.yaml` holds
text only; numbers in it are written as `{metric.model.condition}` placeholders filled from
the CSVs.

## Data-leakage check

Reads `docs/results/*.csv` and the saved logs in `results/final/` only. No episode is run, no
`--final`, no training, no calibration. Nothing is selected from the test numbers: the poster
reports the same rows as the report.

## Tests

`tests/eval/test_poster.py`:
- table cells and headline numbers equal the values in `test_main.csv` / `test_robustness.csv`
  (same rounding as the report macros);
- every `{...}` placeholder in `content.yaml` resolves; no unresolved brace is left in the
  built slide text;
- the built file has exactly one slide of 13.33 × 7.5 in, and every shape lies inside it;
- no "we" / "I" / "our" in the poster text.

## Steps

1. `pip install python-pptx` in `.venv`; pin it in `requirements.txt`.
2. Write `poster/content.yaml` (first full text draft).
3. `src/trafficrl/eval/poster.py` + tests (numbers and placeholders).
4. `scripts/build_poster.py`: header, three columns, bottom strip, native table.
5. Poster figures; place them.
6. Build → LibreOffice PDF → PNG → inspect; fix overflow, overlaps, spacing. Repeat until clean.
7. User review of the rendered poster (text and look); apply changes in `content.yaml`.
8. README, ROADMAP 9.2 + Decision Log, CLAUDE.md, worklog 026; `pytest`, `ruff check`.

## Status (15 Nov)

Steps 1–6 and 8 done; step 7 (user review) open.

**Deviations from the plan**
- **Figure 1** is not `report/figures/architecture.png`: at 4 in wide its text is about 4 pt.
  It is a reduced diagram (inference row, Max-Pressure, training, threshold) drawn with native
  shapes in `build_poster.py`, texts in `content.yaml`.
- **Figure 2** is a poster-specific three-panel figure built from `test_robustness.csv`
  (`poster.fig_robustness`), not a re-render of `figures.py` functions: those need the logs in
  `results/final/` and have too many panels. No `poster/figures` beyond `robustness.png`.
- **Ablation** is three extra rows of Table 1, not a separate figure; **gate firing** is panel
  (c) of Figure 2; **held-out city** is a Findings bullet.
- **Hypotheses** are not listed in the left column; they appear with their verdicts in Findings
  (the plan's own fallback for lack of space). The setup strip moved to the left column.
- **Fonts:** Arial only (no Calibri / Cambria-compatible font on this machine).
- **Placeholders** use `{metric@model@condition}` (the keys of the report macros).
- No `poster/assets/` (no logos).

## Verification

```bash
python scripts/build_poster.py --pdf        # poster/poster.pptx, poster/poster.pdf, poster/figures/
pytest tests/eval/test_poster.py
ruff check src tests scripts
```
Expected: one-page PDF, 16:9; visual check of the PNG render shows no clipped or overlapping
text; headline numbers match `docs/results/test_main.md` (UG-MARL 295.4 ± 1.1 s, DQN
292.2 ± 0.6 s, IDQN 307.9 ± 1.4 s, Max-Pressure 320.8 s).

## Experiment log

No rows: no run of any kind.

## Risks / open questions

- **Too much content for one slide.** Priority if space runs out: keep Table 1, the
  architecture figure, the robustness figure and the findings; drop first the gate-firing plot,
  then the hypotheses lines (they are restated in the findings), then Algorithm 1.
- **Font differences.** LibreOffice renders Calibri with Carlito (same widths); the final check
  should still be done once in PowerPoint by the user.
- **Equation rendering.** The gate rule is set as Unicode text (σ, τ, ≤); if it looks poor it
  is rendered from LaTeX to PNG instead.
- **"No AI contents"** in the format document applies to the poster text as to the report; the
  user reviews and owns the text (same decision as for the report).
- Header details are taken from the report title block (Roll No. 2401AI28); the user confirms
  name, department and e-mail on review.

## Worklog

Executing this plan ends with `worklog/026-2026-11-15-poster.md` and its row in
`worklog/README.md`.
