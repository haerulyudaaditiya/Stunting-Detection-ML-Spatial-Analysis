# Repository guidance

This repository supports ICDEES 2026 paper 1571345184: provincial clustering, stunting classification, and HAZ/WAZ regression. Read `README.md` for the notebook roles and directory layout.

## Working with research files

- Inspect the relevant code, notebook cells, and saved evidence before proposing a change. Check references before moving or deleting files.
- Preserve the six notebook filenames and locations. Supporting analyses read some notebooks and their outputs directly.
- Retain original experiments alongside revisions. Distinguish the original results, supplementary checks, and final paper results when explaining changes.
- Keep saved models, fold assignments, reference tables, and verification records needed by the notebooks and audits. Training or regeneration can replace research evidence; run these only within an authorized experiment and use a separate output directory when appropriate.
- Use `results/regression_corrected/model_freeze_manifest.json` to identify the frozen models evaluated externally. The dashboard export manifest describes browser assets and may represent an earlier research stage.
- The author normally edits the manuscript manually. Provide one precise wording change at a time unless direct document editing is requested.

## Dashboard work

- When available locally, read the relevant approved spec and plan under `docs/superpowers/` before implementation. These planning files are intentionally excluded from Git; otherwise, follow the scope agreed with the user.
- Preserve the three research tasks. Provide the model choices required by the approved dashboard plan and identify the model actually used in each result.
- Associate metrics with their model and evaluation protocol. Jeneponto metrics for frozen Random Forest models must not be attributed to other selectable models.
- Keep the map based on local geographic assets and disclose missing province geometry accurately.
- Reuse the existing interface conventions. Check responsive layout, keyboard use, accessible labels, focus, and contrast when changing the UI.

## Repository maintenance

- Maintain one root `README.md` and one `requirements.txt`. Keep documentation concise and based on files and behavior that actually exist.
- Respect `.gitignore` for local conference documents, scratch files, tool settings, and detailed outputs. Review its exceptions before changing artifact tracking.
- Report what changed and what was verified. Distinguish static inspection, automated checks, browser checks, and unperformed checks.
- Commit, push, publish, or change GitHub settings only within the user's explicit authorization.

## Checks

Run relevant checks for the files changed. Python tests use `python -m unittest discover -s tests -v`; browser prediction tests use `node --test tests/test_predictor.js`. For documentation changes, inspect the diff and run `git diff --check`. Do not rerun full experiments merely to check documentation or styling.
