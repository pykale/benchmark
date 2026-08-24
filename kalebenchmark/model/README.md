# Model modules

Predictors for the `predict` stage. The benchmark's model stage covers what PyKale splits between
`kale.predict` (the network) and `kale.pipeline` (the trainer that fits it), because a benchmark
selects the two together and runs them as one component.

Estimators that already expose `fit`/`predict`, such as scikit-learn's, need no module here: the
shortcuts resolve to those classes directly.

A wrapper here should take the network as an argument rather than construct one, because batching
and training are usually shared across a discipline's architectures while construction is not.
`materials/crystal_gnn.py` follows that: one regressor, plus a small builder per architecture, and
any `torch.nn.Module` from anywhere else works unchanged.

## Disciplines

Discipline-specific modules live in a folder per discipline (`materials/` today; `protein/`,
`cancer/` and others later), so a maintainer only reads the folder they own. Modules that apply
to any discipline stay at this level.
