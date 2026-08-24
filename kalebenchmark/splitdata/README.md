# Data splitting modules

Partitioning for the `splitter` stage: decide which records are used for training and which are
held out. Random splitting delegates to `kale.loaddata.dataset_access.split_by_ratios`.

PyKale keeps its split utilities inside `kale.loaddata`. The benchmark gives them their own stage
because comparability between submissions depends on which split was used, so a task card must be
able to fix it.

`dataset_split` holds the two protocols that need no discipline:

- `RandomSplit(ratios, seed)` — delegates to `split_by_ratios`.
- `PredefinedSplit(partitions, key)` — any protocol defined by listing which records belong to
  which partition: out-of-distribution splits, one outer fold of a cross-validation,
  leave-one-group-out, or a published train/test list. Members are positional indices, or values
  of an identifying field (`mpids`, an accession, a patient id) when `key` is given. It takes no
  arguments a registry could guess, so it is used as an object rather than by name.

A discipline needs no class of its own: the stage accepts any callable taking the records, so a
discipline module is a function supplying its published paths and identifier field; see
`materials/bandgap_split.py`. Subclass `PredefinedSplit` and override `members()` only when the
lists come from somewhere `JsonSplit` cannot read.

## Disciplines

Discipline-specific modules live in a folder per discipline (`materials/` today; `protein/`,
`cancer/` and others later), so a maintainer only reads the folder they own. Modules that apply
to any discipline stay at this level.
