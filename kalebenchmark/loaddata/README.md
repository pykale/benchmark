# Data loading modules

Dataset access for the `dataset` stage: get the measurements from somewhere, as identifiers and
targets only. The representation is chosen later, by `prepdata` and `embed`, so one dataset can
be compared under several encodings.

`json_datasets` reads the `{identifier: {property: value}}` shape that measurements are usually
published in. Both the identifier field and the property are arguments, so one loader serves any
discipline, and one dataset can be benchmarked on any of the properties it publishes. Downstream
stages take the identifier and the target as the frame's first two columns rather than by name.

A discipline needs no class of its own here: the stage accepts any callable returning the
records, so a discipline module is a function supplying its published paths and identifier field.
Use `functools.partial` to fix a different property.

Counterpart of `kale.loaddata`.

## Disciplines

Discipline-specific modules live in a folder per discipline (`materials/` today; `protein/`,
`cancer/` and others later), so a maintainer only reads the folder they own. Modules that apply
to any discipline stay at this level.
