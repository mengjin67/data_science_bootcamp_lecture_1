# CLAUDE.md

Data science bootcamp lecture: auto insurance claim-cost model (synthetic data).

## Layout
- `analysis_pipeline/` notebooks 1-4: ETL -> exploration -> model selection -> deployment
- `analysis_tool_chest/` helper modules imported by the notebooks
- `.local/` generated data and outputs (gitignored)
- `README.md` student setup and overview; `data_science_bootcamp_lecture_1_class_project_intro.md` class project intro (read before the notebooks)

## Commands
- Setup: `uv sync`
- Run a notebook headless, from `analysis_pipeline/`:
  `uv run jupyter nbconvert --to notebook --execute --output-dir <tmp_dir> <notebook>.ipynb`
- Never use `pip`; add packages with `uv add`

## Conventions
- Run notebooks in order 1 -> 4; each reads the previous one's output from `.local/`
- Notebook paths are relative to `analysis_pipeline/`
- Red "Task N" cells are student exercises: edit typos only
- Keep notebook edits to comments/markdown unless asked; do not change saved outputs
- Python 3.10 (`.python-version`)
