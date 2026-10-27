# 005 — Repo name, line endings, commit attribution
**Date:** 2026-10-27   **Tag:** setup   **ROADMAP:** —   **Status:** done

## What
Set three repository conventions:
- **Repo name:** `Agentic-Traffic-Signal-Control`, added to `README.md`.
- **Line endings:** forced to LF everywhere.
- **Commit attribution:** no AI co-author lines in commits or PRs.

## Why
- **Name:** matches PS16's "Agentic AI" wording and the application area.
- **LF endings:** the repo is edited on Windows and cloned into WSL and native Ubuntu.
  Mixed CRLF/LF line endings cause noisy diffs and break shell scripts on Linux.
- **Attribution:** the user is the sole author of the project.

## How
- **`.gitattributes`:** `* text=auto eol=lf`. PDF and PNG files are marked binary.
- **`README.md`:** created with the repo title.
- **No-attribution rule:** added to `CLAUDE.md` Notes and to `.claude/commands/commit.md`
  (step 6). Also saved in Claude's memory.

## Result
All three conventions are in place.

Open issue: `README.md` was saved as UTF-16, so git treats it as a binary file and its diffs
aren't readable. It should be re-saved as UTF-8 when the README is written (ROADMAP 0.1b / 9.4).

## Files / commits
`.gitattributes` · `baf0ae3` — `README.md` · `ec64dce` — `CLAUDE.md`,
`.claude/commands/commit.md` · `9622ac2`

## Next
Re-save `README.md` as UTF-8 when adding the install steps.
