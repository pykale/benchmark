# kalebenchmark

A thin benchmark wrapper around PyKale. PyKale does the machine learning; `kalebenchmark` selects,
connects and runs its components.

```python
from examples.realmat_bag import RealMatBaG

results = RealMatBaG(
    dataset="experimental_bg",
    splitter="random_split",
    prepdata="cif",
    embed="crystal_features",
    predict="svr",
    evaluate=["mae", "mrae"],
).run()
```

Read it left to right: which data, how split, how prepared, how represented, how predicted, how
evaluated, how interpreted.

> **Alpha release.** The API may still change between versions.

## Install

```bash
pip install kalebenchmark
```

The install dependency of the reusable package is only `pykale`.

RealMat-BaG lives under `examples/realmat_bag/`, outside the packaged pipeline. To run that
example, clone this repository, install it with the scientific dependencies, and obtain the
public data/model checkout:

```bash
git clone https://github.com/pykale/benchmark.git
cd benchmark
pip install -e ".[dev]"
pip install numpy pandas scikit-learn pymatgen shap
git clone https://github.com/Shef-AIRE/bandgap-benchmark
cd bandgap-benchmark && unzip cif_file.zip && cd ..
```

They are looked for at `./bandgap-benchmark`; set `REALMAT_BAG_ROOT` to point elsewhere. Nothing
is read from there until a run needs it, and anything that does fails with these instructions.

## Stages

```
dataset -> splitter -> prepdata -> embed -> predict -> prediction
                                                       |- evaluate
                                                       `- interpret
```

Preparation and embedding are fitted on the training partition only. The shortcuts below belong
to `RealMatBaG`; the base `Benchmark` provides `random_split` and `predefined_split`.
Pass `PredefinedSplit(partitions=...)` as an object to supply membership lists.

| Stage | Built-in names | Your own object |
|---|---|---|
| `dataset` | `experimental_bg` | `load()`, or any callable returning the records |
| `splitter` | `random_split`, `predefined_split`, `feature_ood_split` | `split(data)`, or any callable, returning a mapping with `"train"` |
| `prepdata` | `cif` | `transform(partition)`, plus `fit(train)` if it learns anything |
| `embed` | `identity`, `crystal_features` | same as `prepdata` |
| `predict` | `svr`, `random_forest`, `linear_regression`, `cgcnn` | `fit(x, y)`, `predict(x)` |
| `evaluate` | `mae`, `mse`, `mrae`, `r2` | callable `(y_true, y_pred)` |
| `interpret` | `shap`, `permutation_importance` | `interpret(**context)` |

Names are shortcuts, not a permission list: any object with the right methods works in any stage,
with no registration, and `predict`, `evaluate` and `interpret` also take lists.

```python
results = RealMatBaG(
    prepdata=MyDescriptor(),
    predict=SVR(C=10),
    evaluate=["mae", MyMetric()],
).run()
```

An interpreter is given whichever parts of the run it declares: `model`, `data`, `splits`,
`representations`, `predictions`, `targets`, `evaluations`.

## Official tracks

A `TaskCard` fixes only what a leaderboard needs, and leaves the modelling open:

```python
benchmark = RealMatBaG.from_task("experimental_bg_ood", prepdata="cif", predict="svr")
payload = benchmark.run().submit()
```

| Task | Fixed | Open |
|---|---|---|
| `experimental_bg` | dataset, random split, `mae` + `mrae` | prepdata, embed, predict, interpret |
| `experimental_bg_ood` | dataset, feature-OOD split, `mae` + `mrae` | prepdata, embed, predict, interpret |

Eligibility follows the construction path: `RealMatBaG(...)` explores and cannot submit,
`RealMatBaG.from_task(...)` is official and can. Passing a fixed stage raises. `submit()` sends
the predictions the run already produced; it never predicts again.

## Layout

The installed package is domain-neutral. Concrete benchmarks live outside it as examples:

```
kalebenchmark/
├── benchmark.py    Benchmark base class and the domain-neutral registry
├── pipeline.py     stage execution
├── task.py
├── results.py
└── splitdata/      PyKale-backed reusable splitting

examples/
└── realmat_bag/    data access, scientific components, shortcuts and TaskCards
```

A concrete example is a `Benchmark` subclass carrying its own `BUILTINS` and `TASKS`, and
overriding `_execute` only if its protocol differs. RealMat-BaG and its optional dependencies are
therefore not imported or installed with the core benchmark package.

## Examples and tests

Run these commands from the repository root:

```bash
python -m examples.simple_benchmark.main     # every stage by name
python -m examples.custom_components.main    # custom representation, model and metric
python -m examples.official_task.main        # task card and submission
python -m examples.crystal_gnn.main          # CGCNN predictor

pytest              # the materials tests skip without the checkout
pytest -m slow      # only the CIF and CGCNN integration tests
```
