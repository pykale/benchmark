# Embedding modules

Representations for the `embed` stage: turn prepared data into the features a model is trained
on. A component that learns from the data adds `fit`, which is called on the training partition only; one that does not needs `transform` alone.

This stage is where the classical and graph paths diverge: a featuriser produces a feature matrix
for scikit-learn estimators, while the no-op representation keeps the graphs for models that
consume them directly.

Counterpart of `kale.embed`.

## Disciplines

Discipline-specific modules live in a folder per discipline (`materials/` today; `protein/`,
`cancer/` and others later), so a maintainer only reads the folder they own. Modules that apply
to any discipline stay at this level.
