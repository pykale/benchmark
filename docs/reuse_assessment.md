# Source-code reuse assessment

`kalebenchmark` composes existing implementations. This file records, stage by
stage, what was reused and what was written here, so that duplication can be
reviewed.

## Where each implementation comes from

Two sources are used, and they are kept separate on purpose:

- **Released PyKale** — only APIs that exist in the published package.
- **The RealMat-BaG checkout** (`bandgap-benchmark/`, `Shef-AIRE/bandgap-benchmark`)
  for the materials-specific work, which is *not* in released PyKale yet. These
  imports are function-local and marked `TODO(pykale)`; see
  [Migration back to PyKale](#migration-back-to-pykale).

## Package layout

Two levels: one subpackage per stage, mirroring PyKale's classification, and one folder per
discipline inside it, so a discipline maintainer reads only the folders they own.

| Subpackage | Stage argument | PyKale counterpart |
|---|---|---|
| `kalebenchmark/loaddata/` | `dataset` | `kale.loaddata` |
| `kalebenchmark/splitdata/` | `splitter` | split utilities inside `kale.loaddata` |
| `kalebenchmark/prepdata/` | `prepdata` | `kale.prepdata` |
| `kalebenchmark/embed/` | `embed` | `kale.embed` |
| `kalebenchmark/model/` | `predict` | `kale.predict` + `kale.pipeline` |
| `kalebenchmark/evaluation/` | `evaluate` | `kale.evaluate` |
| `kalebenchmark/interpretation/` | `interpret` | `kale.interpret` |
| `kalebenchmark/utils/` | -- | `kale.utils` |
| `kalebenchmark/benchmarks/` | -- | concrete `Benchmark` subclasses, one folder per discipline |

Splitting gets its own subpackage because comparability between submissions depends on which
split was used, so a task card must be able to fix it independently of the dataset. The model
subpackage merges PyKale's `predict` and `pipeline` because a benchmark selects the network and
the trainer that fits it as one component.

Within a stage, a module that applies to any discipline stays at the stage level
(`splitdata/dataset_split.py`, `interpretation/feature_attribution.py`), and everything else goes
in a discipline folder (`materials/` today; `protein/`, `cancer/` later). `Benchmark` itself
knows no discipline: its registry holds only domain-neutral shortcuts, and each discipline's
subclass carries its own `BUILTINS` and `TASKS`. Adding a discipline therefore touches no shared
file.

A stage folder holds no code when nothing needs wrapping: the shortcuts for scikit-learn
estimators and metrics resolve to those libraries directly.

A discipline writes a class only where a stage has to carry state between calls — `prepdata`,
`predict` and `interpret`. The `dataset`, `splitter` and `evaluate` stages accept any callable,
so those are plain functions, configurable with `functools.partial`.

## Stage by stage

### Splitting

```
Random split
→ reused kale.loaddata.dataset_access.split_by_ratios
→ reused kale.utils.seed.set_seed for reproducibility (split_by_ratios takes no
  generator argument and draws from the global torch RNG)
→ no split algorithm implemented; RandomSplit only maps the returned
  Subset.indices back onto the input rows
```

```
Predefined splits (out-of-distribution, outer folds, leave-one-group-out)
→ one domain-neutral PredefinedSplit: a protocol is a list of which records belong to which
  partition, by positional index or by an identifying field
→ selection is a pandas isin(), or a membership test for records that are not a frame
→ no partitioning heuristic implemented

RealMat-BaG feature out-of-distribution split
→ reused the shipped data/splits_feature_ood/*.json identifier lists
→ new code: feature_ood_split, a function naming those files and key="mpids". The chemsys,
  crystalsys, periodic-group and leave-one-material-out regimes need no code at all: they are
  the same protocol with different files, so JsonSplit takes them directly
```

### Dataset

```
experimental_bg
→ reused the shipped data/fine_tune/*.json measurements (1,705 materials)
→ new code: JsonRecords, ~10 lines of json -> DataFrame I/O, with the identifier field and the
  property both arguments. The reference repository's equivalent (load_json_as_dataframe) lives
  inside its main.py script and is not importable, and released PyKale has no load_json_to_df
→ the dataset yields identifiers and targets only, so the representation stays a separate,
  replaceable decision, and downstream stages read the first two columns rather than fixed names
```

### Preparation

```
cif
→ reused kale.loaddata.materials_datasets.CIFData
→ no neighbour search, Gaussian expansion or atom initialisation implemented
```

### Embedding

```
crystal_features
→ reused kale.prepdata.materials_features.extract_features
→ no feature aggregation implemented

identity
→ reused sklearn.preprocessing.FunctionTransformer, which is already a
  dependency and is a true no-op (verified: fit returns self, transform returns
  the same object)
→ no passthrough class written. kale.embed.image_cnn.Identity was considered and
  rejected: it is an nn.Module with forward(), not a fit/transform stage
```

### Prediction

```
svr, random_forest, linear_regression
→ reused sklearn.svm.SVR, sklearn.ensemble.RandomForestRegressor and
  sklearn.linear_model.LinearRegression directly as registry values
→ no wrapper classes. An earlier prototype had SVRPredictor /
  RandomForestPredictor / LinearRegressionPredictor; they added nothing over the
  estimators they held and were deleted

cgcnn
→ reused realmat_bag.pipeline.models.cgcnn.CGCNN.CrystalGraphConvNet (network),
  realmat_bag.pipeline.trainer.MaterialsTrainer (loss, metrics, optimisers),
  realmat_bag.loaddata.collate.collate_crystal_batch (batching),
  kale.loaddata.materials_datasets.CIFData.collate_fn (batching) and
  pytorch_lightning.Trainer (training loop)
→ new code: CrystalGraphRegressor, which constructs those objects, reads the
  feature dimensions from the first sample, and exposes fit/predict. It contains
  no loss, no metric and no optimisation step. Its only non-trivial glue is a
  six-line inference loop, needed because MaterialsTrainer defines no
  predict_step
```

### Evaluation

```
mae, mse, r2
→ reused sklearn.metrics.mean_absolute_error, mean_squared_error and r2_score
  directly as registry values
→ an earlier prototype hand-rolled MAE with numpy; that was deleted

mrae
→ reused realmat_bag.pipeline.models.classical_ml.mean_relative_error
→ new code: a thin tensor/signature adapter for the benchmark's
  scikit-learn-style (y_true, y_pred) convention
```

### Interpretation

```
shap
→ reused shap.Explainer
→ released PyKale has no SHAP support; kale.interpret provides only captum-based
  signal/image attribution and plotting helpers

permutation_importance
→ reused sklearn.inspection.permutation_importance
```

### Execution

```
Pipeline
→ kalebenchmark.pipeline only dispatches: it calls load / split / fit /
  transform / fit / predict / metric / interpret on the selected components
→ preprocessing and embedding are fitted on the training partition only, so a
  data-dependent step cannot leak the evaluation partition
→ no trainer, no loop, no framework of its own. PyTorch models are trained by
  Lightning through the component that owns them
```

## Considered and deliberately not used

| Existing API | Why it was not used |
|---|---|
| `kale.evaluate.cross_validation.leave_one_group_out` | Classification only: it one-hot encodes groups and scores with `accuracy_score`. Band-gap prediction is regression. |
| `kale.evaluate.cross_validation.cross_validate` | Usable, but this deliverable runs a single split per benchmark; k-fold is out of scope. |
| `kale.pipeline.mpca_trainer.MPCATrainer` | A classifier, and tied to tensor-shaped inputs. |
| `kale.embed.image_cnn.Identity` | An `nn.Module`; the embed stage needs `fit`/`transform`. |
| `kale.prepdata.tabular_transform.ToTensor` | Call-only transform for PyG-style datasets; nothing in this pipeline needs it. |

## Remaining gaps

PyKale supplies the material loading, featurisation, CGCNN, regression trainer and MRE metric.
Scikit-learn supplies classical predictors and its regression metrics; SHAP supplies SHAP
interpretation. The RealMat-BaG checkout is data only and may be replaced by explicit paths.

`tests/test_reuse.py` patches each reused implementation and asserts it is
called, so an accidental re-implementation fails the suite rather than silently
diverging.
