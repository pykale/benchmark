# Evaluation modules

Metrics for the `evaluate` stage, called as `metric(y_true, y_pred)`.

Only thin adapters needed to match the benchmark calling convention live here; implementations
are reused from PyKale or scikit-learn where available.

Counterpart of `kale.evaluate`.

## Disciplines

Discipline-specific modules live in a folder per discipline (`materials/` today; `protein/`,
`cancer/` and others later), so a maintainer only reads the folder they own. Modules that apply
to any discipline stay at this level.
