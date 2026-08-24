# Interpretation modules

Interpreters for the `interpret` stage. An interpreter declares the parts of the run context it
needs, out of `model`, `data`, `splits`, `representations`, `predictions`, `targets` and
`evaluations`, and is given exactly those.

Counterpart of `kale.interpret`.

## Disciplines

Discipline-specific modules live in a folder per discipline (`materials/` today; `protein/`,
`cancer/` and others later), so a maintainer only reads the folder they own. Modules that apply
to any discipline stay at this level.
