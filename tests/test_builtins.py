"""Tests for the domain-neutral shortcuts carried by the base benchmark.

Discipline-specific shortcuts are tested by that discipline, under ``tests/<discipline>/``.
"""

import numpy as np
import pytest

from kalebenchmark import Benchmark


@pytest.mark.parametrize("stage", sorted(Benchmark.BUILTINS))
def test_every_generic_shortcut_resolves(stage):
    for name in Benchmark.BUILTINS[stage]:
        assert Benchmark.resolve(stage, name) is not None


def test_identity_embedding_is_a_no_op():
    class FakeDataset:
        def __len__(self):
            return 3

        def __getitem__(self, index):
            return index

    embedding = Benchmark.resolve("embed", "identity")
    dataset = FakeDataset()
    assert embedding.fit(dataset) is embedding
    assert embedding.transform(dataset) is dataset
    features, targets = embedding.transform((np.zeros((3, 2)), np.zeros(3)))
    np.testing.assert_array_equal(features, np.zeros((3, 2)))
    np.testing.assert_array_equal(targets, np.zeros(3))


def test_sklearn_estimators_are_used_directly():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.svm import SVR

    assert Benchmark.BUILTINS["predict"]["svr"] is SVR
    assert Benchmark.BUILTINS["predict"]["random_forest"] is RandomForestRegressor
    assert Benchmark.BUILTINS["predict"]["linear_regression"] is LinearRegression


def test_builtin_pipeline_runs_on_tabular_features():
    class TabularDataset:
        def load(self):
            rng = np.random.default_rng(0)
            features = rng.normal(size=(40, 3))
            return features, features @ [1.0, -2.0, 0.5]

    class TabularSplitter:
        def split(self, data):
            features, targets = data
            return {"train": (features[:30], targets[:30]), "test": (features[30:], targets[30:])}

    results = Benchmark(
        dataset=TabularDataset(),
        splitter=TabularSplitter(),
        embed="identity",
        predict="linear_regression",
        evaluate=["mae", "r2"],
    ).run()
    assert results.evaluations["mae"] == pytest.approx(0.0, abs=1e-6)
    assert results.evaluations["r2"] == pytest.approx(1.0)


def test_the_torch_runner_is_domain_neutral():
    """TorchRegressor implements no loss, no batching and no network: all three are injected."""
    from kalebenchmark.model.torch_trainer import TorchRegressor

    runner = TorchRegressor()
    assert runner.collate() is None
    with pytest.raises(ValueError, match="needs a module"):
        runner.build_module(object())
    with pytest.raises(RuntimeError, match="called before fit"):
        runner.predict([])
