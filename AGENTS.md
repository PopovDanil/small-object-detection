# AGENTS.md

Instructions for AI coding assistants working in this repository. The goal is code that
looks like it was written by a careful human maintainer, not generated filler.

## Project

Small-object detection on VisDrone2019-DET: compare 640 full-image, 1280 full-image and
tiled inference with YOLO26m/YOLO26x. Python 3.12, managed with `uv`.

- `src/evaluation/` — shared evaluation code, especially `metrics.py` (`evaluate_split`)
- `scripts/` — non-notebook entry points (data prep, Kaggle, results packing)
- `notebooks/` — experiment stages; they orchestrate, they do not own shared logic
- `docs/`, `results/` — reports and generated outputs

Read `README.md` before changing the evaluation protocol, data layout or run structure.
Do not change protocol constants, metrics or split usage unless explicitly asked.

## Commands

```bash
uv sync                          # install dependencies
uv run ruff check . --fix        # lint and autofix
uv run ruff format .             # format
uv run mypy src                  # type check
uv run pre-commit run --all-files
```

Always run the relevant checks after editing Python or notebook code. Do not introduce
new dependencies without need; prefer what is already in `pyproject.toml`.

## Comments

Write comments only when the code cannot explain itself.

Do not add comments that restate the next line:

```python
# BAD
# Load the labels.
labels = load_labels(path)

# GOOD
labels = load_labels(path)
```

Comments must explain *why*, a non-obvious invariant, or a deliberate trade-off:

```python
# GOOD: one shared cap; Ultralytics' per-forward-pass max_det would otherwise
# cap tiled inference per tile and full-image inference per image.
MAX_DET = 1000
```

Do not narrate obvious control flow (`# loop over images`, `# return result`,
`# check if file exists`), label every block with banner comments, or leave
commented-out code. Delete dead code instead of commenting it out.

## No cross-notebook or cross-cell references in code

Code must stand alone. Never write comments or docstrings such as:

- `# notebook 6 does this, so use the identical structure here`
- `# same as Cell 12 in baseline.ipynb`
- `# copied from highres.ipynb`
- `# see the tiling notebook for the real implementation`

If logic is shared, move it into an importable module under `src/` and call it from
both places. If a notebook stage must run in a specific order, document that once in
`README.md`, not in code comments or cell-by-cell instructions.

Notebook markdown may describe the experiment stage, but must not reference cell numbers,
"notebook 6", or tell the reader to copy code from another notebook.

## Docstrings

A docstring is not required for every function. Use one when it adds information a
reader cannot get from the name, signature and type hints.

- Required: public modules and non-trivial public functions (`evaluate_split`, data
  preparation entry points) where the contract, units, metric definitions or side
  effects are not obvious.
- Allowed: a short module docstring that explains the purpose and any caveat.
- Not allowed: `"""Load the labels."""`, `"""Return the path."""`, restating parameter
  types, listing obvious parameters, or boilerplate sections with no content.

Keep docstrings short and factual. No marketing language, no "This function is
responsible for...", no step-by-step duplication of the body.

## Formatting and spacing

- Follow `ruff format`; do not hand-wrap code into dense one-liners to save lines.
- Use blank lines between logical blocks inside functions. Long functions with no
  visual grouping are a smell; split them or add separation, not banner comments.
- One statement per line. No semicolon-chained statements, no nested one-line
  `if ...: return ...` pyramids.
- Standard import grouping (`ruff`/isort): standard library, third-party, local.
- Use `pathlib` for paths, f-strings for formatting, explicit type hints at module
  boundaries.
- Avoid `Any` except at genuinely untyped library boundaries. Avoid bare `except`.
- Match the surrounding style; do not reformat untouched code unrelated to the task.

## Python and notebooks

- Shared behavior belongs in `src/` or `scripts/`; notebooks call it. Do not duplicate
  a function in a notebook because it is convenient.
- Keep notebook cells focused. Do not add "setup", "imports", "helpers" cells that
  repeat code already present in the repo.
- Every experiment must use `evaluate_split` from `src/evaluation/metrics.py`. Do not
  add a second evaluator.
- Keep seed 42, fixed confidence/IoU values, test-dev reporting and validation-only
  checkpoint selection unless the task explicitly changes the protocol.
- Notebook outputs are not source; do not commit generated data, weights or run
  artifacts.

## Git workflow

Use Conventional Commits and short-lived branches off `main`.

Branch names mirror the commit type:

```text
feat/tiling-overlap-sweep
fix/highres-runtime-sync
docs/agents-guidelines
refactor/shared-evaluator
```

Keep one logical change per commit. Do not mix formatting, refactoring and behavior
changes in the same commit.

Commit messages:

```text
<type>(<scope>): <summary>

<optional body: why the change is needed, wrapped at 72 columns>
```

- Types: `feat`, `fix`, `docs`, `refactor`, `test`, `perf`, `build`, `ci`, `chore`.
- Scopes are optional but useful here: `eval`, `data`, `training`, `inference`,
  `notebooks`, `results`, `scripts`, `docs`.
- Summary: imperative mood, lowercase, no trailing period, at most 72 characters.
- Body: explain motivation and trade-offs, not a line-by-line restatement of the diff.
- Breaking changes: add `!` after the type/scope and a `BREAKING CHANGE:` footer.
- No generated-by or AI-attribution trailers, no emoji, no vague summaries such as
  "update code" or "fix stuff".

Examples:

```text
feat(eval): share MAX_DET across full-image and tiled inference
fix(tiling): clip merged boxes to original image bounds
docs(results): report overlap-sweep runtime
```

Before committing, inspect `git diff` and stage only the intended files. Do not commit
generated data, weights, run artifacts or notebook outputs. Do not amend, force-push or
commit unless explicitly asked.

## Assistant behavior

- Prefer the smallest correct change. Do not add abstractions, wrappers, config flags or
  defensive checks for scenarios that cannot happen.
- Do not add comments, docstrings, README sections or tests beyond what the task needs.
- Do not claim something was run or verified unless it was.
- Do not commit unless asked.
