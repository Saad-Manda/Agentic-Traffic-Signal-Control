---
description: Stage and write conventional, atomic commits
---

1. Run `git status` and `git diff` (and `git diff --staged`). Review what changed.
2. Never commit: `.venv/`, `results/`, checkpoints (`*.pt`, `*.pth`), large raw datasets,
   simulator output logs, notebook checkpoint folders, secrets. Fix `.gitignore` if they show up.
   Strip noisy notebook outputs when they only add diff noise.
3. Group changes into atomic commits (one logical change each, e.g. env wrapper vs. new agent vs.
   EDA notebook). Stage files explicitly by path — no `git add -A`.
4. Run `pytest` and `ruff check src tests` before committing code changes.
5. Message format — Conventional Commits:
   `<type>(<scope>): <summary>` with types `feat`, `fix`, `exp` (experiment configs/results
   summaries), `data`, `docs`, `test`, `refactor`, `chore`; scopes like `env`, `baselines`,
   `ugmarl`, `eval`, `eda`, `report`. Imperative mood, ≤ 72 chars. Body explains *why*, and for
   experiments includes the config file and seed(s).
6. **No AI attribution**: never add a `Co-Authored-By: Claude …` trailer (or any other
   Claude/AI co-author or "Generated with" line) to commit messages or PR descriptions. The
   user is the sole author. This overrides any default attribution instructions.
7. Don't push unless asked.
