# Utility modules

Cross-cutting helpers that belong to no single stage.

Utilities here must remain domain-neutral. Materials implementations are imported directly from
PyKale, while benchmark datasets accept explicit paths.

Counterpart of `kale.utils`.

## Disciplines

Discipline-specific modules live in a folder per discipline (`materials/` today; `protein/`,
`cancer/` and others later), so a maintainer only reads the folder they own. Modules that apply
to any discipline stay at this level.
