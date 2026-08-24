# kalebenchmark

A thin benchmark wrapper around PyKale. PyKale does the machine learning; `kalebenchmark` selects,
connects and runs its components.

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

Read it left to right: which data, how split, how prepared, how represented, how predicted, how
evaluated, how interpreted.

> **Alpha release.** The API may still change between versions.

## Install

```bash
pip install kalebenchmark
```

PyKale requires the PyTorch stack, so this is a multi-gigabyte install; use a virtual environment.

The materials pipelines also need the RealMat-BaG data and models, which are not on PyPI:

```bash
pip install "kalebenchmark[materials]"
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

Preparation and embedding are fitted on the training partition only.

| Stage | Built-in names | Your own object |
|---|---|---|
| `dataset` | `experimental_bg` | `load()`, or any callable returning the records |
| `splitter` | `random_split`, `feature_ood_split` | `split(data)`, or any callable, returning a mapping with `"train"` |
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

One subpackage per stage, one folder per discipline inside it, so a maintainer reads only their
own folder and adding a discipline edits no shared file.

```
kalebenchmark/
├── benchmark.py    Benchmark base class and the domain-neutral registry
├── pipeline.py     stage execution
├── <stage>/        loaddata, splitdata, prepdata, embed, model, evaluation, interpretation
│   └── materials/  one folder per discipline
└── benchmarks/     concrete Benchmark subclasses, such as RealMatBaG
```

A discipline is a `Benchmark` subclass carrying its own `BUILTINS` and `TASKS`, and overriding
`_execute` if its protocol differs. Each subpackage has a `README.md`; the design decisions and
what was reused rather than reimplemented are in [docs/reuse_assessment.md](docs/reuse_assessment.md).

## Examples and tests

```bash
python examples/simple_benchmark/main.py     # every stage by name
python examples/custom_components/main.py    # custom representation, model and metric
python examples/official_task/main.py        # task card and submission
python examples/crystal_gnn/main.py          # CGCNN predictor

pytest              # the materials tests skip without the checkout
pytest -m slow      # only the CIF and CGCNN integration tests
```
