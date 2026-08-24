# Data preparation modules

Preparation for the `prepdata` stage: turn records into the raw structure a model can consume,
such as crystal graphs built from CIF files. A component that learns from the data adds `fit`, which is called on the training partition only; one that does not needs `transform` alone.

Counterpart of `kale.prepdata`.

## Disciplines

Discipline-specific modules live in a folder per discipline (`materials/` today; `protein/`,
`cancer/` and others later), so a maintainer only reads the folder they own. Modules that apply
to any discipline stay at this level.
