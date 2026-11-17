# 028 — Report appendix and submission folder (results Excel)
**Date:** 2026-11-17   **Tag:** report, submission, decision   **ROADMAP:** 9.1   **Status:** done

## What
- Added an appendix to the report: A.1 code / poster / data links, A.2 reproduction commands,
  A.3 learning curves (Figure A.1).
- Built `submission/results/results.xlsx`: one sheet, one row per model (4 baselines,
  UG-MARL, 3 ablations) with test metrics and the Python files behind each row.

## Why
Submission requirements of 17 Nov: a results folder with one Excel file listing every model
and the proposed model with the Python files that produced the results; an appendix with the
code link (https://github.com/Saad-Manda/PBL-Agentic-AI) and a poster link (to follow).
User decisions: appendix = links + reproduction + learning curves (not hyperparameters,
not extra results, no AI-use statement); Excel = main table only; the code link folder is
made by the user; submission folders under `submission/` in the repo.

## How
- The public repo `PBL-Agentic-AI` was cloned (read-only) to check its layout. It is a
  trimmed version of this repo: no `scripts/final_tables.py` / `plot_training.py`; tables and
  learning curves come from `notebooks/02_results.ipynb`. The appendix commands and the Excel
  file names use only paths that exist there (`scripts/tune.py`, `calibrate_gate.py`,
  `lock_models.py`, `robustness.py`, the agent modules under `src/trafficrl/agents/`).
- The learning-curve figure is `docs/figures/training/learning_curves_phase6.png` (identical,
  by MD5, to `docs/figures/training/learning_curves.png` in the public repo).
- Typed numbers in A.3 were computed from `results/train/*_final_s*/val_log.csv` (first episode
  with the seed mean below Max-Pressure's 326.1 s: DQN 25, −attention 30, −ensemble 30,
  UG-MARL 40, IDQN 45). A first draft said "about 40 episodes" for all; corrected to 25–45.
- Excel: built from `docs/results/test_main.csv`, `test_robustness.csv`, `test_efficiency.csv`,
  `val_phase6.csv` by a one-off script in the session scratchpad (openpyxl installed there,
  not in `.venv`); numbers match `test_main.md` and the report.
- Poster link: macro `\posterlink` in `report/main.tex` prints "[poster link to be added]".

## Result
- `report/report.pdf`: 14 pages (limit 12–15), `check_report.py`: ok, no undefined references;
  ToC lists "A Appendix" with A.1–A.3; List of Figures gains Figure A.1.
- `submission/results/results.xlsx`: 8 model rows; UG-MARL highlighted.
- No simulation, no `--final`, no experiment-log row.
- `dataset_link/dataset.txt` (made earlier this session) was no longer on disk when the
  submission folder was set up; not recreated (user's own folders may hold it).

## Files changed
| File | Action | What changed | Why |
|---|---|---|---|
| `report/chapters/06-appendix.tex` | created | chapter "Appendix": A.1 links (code, poster, CoLight @ `04440b7`), A.2 Figure A.1 + interpretation (reproduction section added, then removed on request) | submission requirement and user's choice A+B+C |
| `report/main.tex` | modified | `\appendix` + `\input{chapters/06-appendix}` after the bibliography; `\newcommand{\posterlink}` with the Canva URL; `../docs/figures/training/` added to `\graphicspath` | include the appendix; poster URL comes later; figure path |
| `report/chapters/01-introduction.tex` | modified | §1.5 Contributions: humanizer pass (opener, item 2 stability sentence, item 3 contrasts, items 4–6 closers) | user request; content unchanged |
| `report/chapters/00-abstract.tex` | modified | removed the `Keywords:` line and the `\medskip` before it | not required by the format document; user request |
| `report/report.pdf` | generated | rebuilt, 13 pages | deliverable |
| `report/checks/claims.md` | modified | three typed-number rows for A.1–A.3 (episodes 25–45, ~10 s IDQN gap, 100 episodes / commits) | every typed number traceable |
| `submission/results/results.xlsx` | created | one sheet "Results": model, type, validation and test travel time, waiting time, max wait, throughput, noise-0.2 travel time, worst-case regret, fallback rate, params, Python files (model / training / evaluation), notes | submission requirement |
| `CLAUDE.md` | modified | report note: appendix added, 14 pages, poster placeholder; removed "no appendix" | keep the session primer correct |
| `worklog/028-2026-11-17-appendix-submission.md`, `worklog/README.md` | created / modified | this entry and its index row | work log rule |

## Revision (same day)
- Poster link received: `\posterlink` = https://canva.link/nj4h08y8a6coxa5 (no trailing
  period after it, so a copied link stays clean).
- User asked to remove the "Result appendix": read as A.2 "Reproducing the Results"
  (the only results section of the appendix); removed. The appendix is now A.1 links,
  A.2 learning curves. Report back to **13 pages**.
- Humanizer pass (user request) on the appendix and on §1.5 Contributions. Removed: the
  self-describing opener ("The major contributions ... are summarised below"), the staged
  "The stability claim is limited to what holds.", three parallel "X, and here Y" contrasts in
  item 3 (now direct statements per prior work), the closer "the negative result is reported
  as it came out", "The limit found there is stated:", bold labels in the appendix link list.
  Kept: the bold category labels in §1.5 (the mentor's contribution categories, worklog/027),
  every number, `\res` macro, citation and cross-reference. Checked that no claim changed:
  first drafts said "stability is proved", "CoLight ... without testing them" and "training
  longer would be unlikely to change the results"; all three over-claimed and were reworded
  back to the original meaning.
- `report/checks/claims.md`: appendix rows renumbered A.3 → A.2; CityFlow-commit row dropped
  (A.2 reproduction section is gone).
- Keywords line removed from the abstract (user request). The format document does not ask
  for keywords (preliminary pages: title, abstract, contents, lists); the line came from the
  14 Nov draft by IEEE / ACM habit, and worklog/024 wrongly attributed it to the format
  document. Report stays at 13 pages, `check_report.py` ok.

## Commits
`a55a104` docs(report): appendix; `665b3c1` docs(submission): results workbook; `82854af` docs(submission): cited papers

## Next
User review of the humanized contributions and appendix; commit.
