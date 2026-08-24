# kalebenchmark

A thin benchmark wrapper around PyKale. PyKale provides the reusable machine-learning
components; `kalebenchmark` selects, connects, constrains and runs them.

```python
from kalebenchmark.benchmarks.materials.bandgap import RealMatBaG

results = RealMatBaG(
    dataset="experimental_bg",
    splitter="random_split",
    prepdata="cif",
    embed="crystal_features",
    predict="svr",
    evaluate=["mae", "mrae"],
).run()
```

The configuration reads left to right: which data, how split, how prepared, how
represented, how predicted, how evaluated, how interpreted.

## Install

```bash
pip install -e .
```

The RealMat-BaG example needs its public data checkout. Crystal processing and models come from
PyKale:

```bash
git clone https://github.com/Shef-AIRE/bandgap-benchmark
cd bandgap-benchmark && unzip cif_file.zip && cd ..
pip install pymatgen torch pytorch-lightning   # or: pip install -e ".[materials]"
```

The default data path is `./bandgap-benchmark`; pass explicit `root`, JSON paths and CIF paths to
use data stored elsewhere. No component modifies `sys.path` or requires a repository named
RealMat-BaG.

## Stages

```
dataset -> splitter -> prepdata -> embed -> predict -> prediction
                                                       |- evaluate
                                                       `- interpret
```

Preprocessing and embedding are fitted on the training partition only, so a
data-dependent step cannot see the evaluation partition.

| Stage | Built-in names (RealMatBaG) | Protocol for your own object |
|---|---|---|
| `dataset` | `experimental_bg` | `load()`, or any callable returning the records |
| `splitter` | `random_split`, `feature_ood_split` | `split(data)`, or any callable, returning a mapping with `"train"` |
| `prepdata` | `cif` | `transform(partition)`, plus `fit(train)` if it learns anything |
| `embed` | `identity`, `crystal_features` | `transform(partition)`, plus `fit(train)` if it learns anything |
| `predict` | `svr`, `random_forest`, `linear_regression`, `cgcnn` | `fit(x, y)`, `predict(x)` |

The `cgcnn` shortcut is one configuration of an architecture-independent regressor: pass
`CrystalGraphRegressor(model=leftnet())`, or any `torch.nn.Module`, to train something else.
| `evaluate` | `mae`, `mse`, `mrae`, `r2` | callable `(y_true, y_pred)` |
| `interpret` | `shap`, `permutation_importance` | `interpret(**context)` |

## Layout

Two levels: one subpackage per stage, following PyKale's classification, and one folder per
discipline inside it. A discipline maintainer only reads the folders they own.

```
kalebenchmark/
├── benchmark.py          Benchmark base class, generic shortcuts, subclass contract
├── pipeline.py           stage execution
├── task.py, results.py
├── benchmarks/           concrete benchmarks   <- materials/bandgap.py -> RealMatBaG
├── loaddata/             dataset stage         <- kale.loaddata
├── splitdata/            splitter stage        <- split utilities in kale.loaddata
├── prepdata/             prepdata stage        <- kale.prepdata
├── embed/                embed stage           <- kale.embed
├── model/                predict stage         <- kale.predict + kale.pipeline
├── evaluation/           evaluate stage        <- kale.evaluate
├── interpretation/       interpret stage       <- kale.interpret
└── utils/                cross-cutting         <- kale.utils

<stage>/
├── README.md
├── <generic>.py          applies to any discipline, e.g. splitdata/dataset_split.py
└── materials/            one folder per discipline; protein/, cancer/ later
    └── <topic>.py        e.g. embed/materials/crystal_features.py
```

`splitdata` is a stage of its own because comparability depends on which split was used, so a
task card must be able to fix it independently of the dataset. `model` merges PyKale's `predict`
and `pipeline`, since a benchmark picks the network and the trainer that fits it as one
component.

### Adding a discipline

`Benchmark` is a base class that knows no discipline: its registry holds only domain-neutral
shortcuts such as `random_split`, `svr` and `mae`. A discipline is a subclass carrying its own
shortcuts and tracks, so adding one never edits shared files:

```python
class RealMatBaG(Benchmark):
    BUILTINS = merge_builtins(GENERIC_BUILTINS, MATERIALS_BUILTINS)
    TASKS = MATERIALS_TASKS
```

1. add `<stage>/<discipline>/` modules for whatever needs wrapping;
2. add `benchmarks/<discipline>/<name>.py` with the subclass, its registry and its task cards;
3. add `tests/<discipline>/`.

Subclasses must keep the seven stage keyword arguments, because `from_task` constructs with
`cls(...)`, and should override `_execute` rather than `run` to change the protocol (for instance
cross-validation), so that later changes to the flow are inherited. `tests/test_subclass.py`
pins both.

Names are shortcuts, not a permission list. Any object with the right methods
works in any stage, with no registration and no change to this package:

```python
results = RealMatBaG(
    dataset=MyDataset(),
    splitter=MySplitter(),
    prepdata=MyPrepData(),
    embed=MyEmbedding(),
    predict=MyPredictor(),
    evaluate=["mae", MyMetric()],
    interpret=MyInterpreter(),
).run()
```

Built-in and custom components mix freely, and `predict`, `evaluate` and
`interpret` accept lists. Configured library objects can be passed directly,
because the shortcuts resolve to the libraries' own classes:

```python
from sklearn.svm import SVR

RealMatBaG(..., predict=SVR(C=10, epsilon=0.05))
```

An interpreter receives whatever part of the run context it declares: `model`,
`data`, `splits`, `representations`, `predictions`, `targets` and `evaluations`.

## Official tracks

A `TaskCard` fixes only what a leaderboard needs for comparability and leaves
the modelling stages open:

```python
benchmark = RealMatBaG.from_task(
    "experimental_bg_ood",
    prepdata="cif",
    embed="crystal_features",
    predict="svr",
)
results = benchmark.run()
payload = results.submit()
```

| Task | Fixed | Open |
|---|---|---|
| `experimental_bg` | dataset, random split, `mae` + `mrae` | prepdata, embed, predict, interpret |
| `experimental_bg_ood` | dataset, feature-OOD split, `mae` + `mrae` | prepdata, embed, predict, interpret |

Eligibility follows the construction path rather than a post-hoc comparison:
`RealMatBaG(...)` is exploration and cannot submit, `RealMatBaG.from_task(...)` is
official and can. Passing a stage the card fixes raises immediately:

```text
ValueError: 'splitter' is fixed by TaskCard 'experimental_bg_ood' and cannot be overridden.
```

`submit()` sends the predictions the run already produced; it never predicts
again. Pass `submitter=` a callable to forward the payload somewhere.

## Examples

```bash
python examples/simple_benchmark/main.py     # every stage by name
python examples/custom_components/main.py    # custom representation, model and metric
python examples/official_task/main.py        # task card and submission
python examples/crystal_gnn/main.py --max-epochs 2 --limit 200   # CGCNN predictor
```

The graph-network example is sized as a smoke test; its accuracy is not
meaningful at two epochs on 200 materials.

## Tests

```bash
pytest              # everything; the materials tests skip without the checkout
pytest -m slow      # only the CIF and CGCNN integration tests
```

The fast tests use in-memory fakes, so they pass with neither torch, pymatgen nor the checkout
installed.

`tests/test_reuse.py` patches each reused implementation and asserts it is
called, so a re-implementation of splitting, featurisation or a metric fails the
suite.
