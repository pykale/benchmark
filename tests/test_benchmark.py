"""Tests for component resolution and pipeline execution."""

import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR

from kalebenchmark.benchmark import Benchmark

from .helpers.fake_components import Predictor


def test_registered_class_is_instantiated_and_function_is_used_as_is():
    assert isinstance(Benchmark.resolve("predict", "svr"), SVR)
    assert Benchmark.resolve("evaluate", "mae") is mean_absolute_error


def test_custom_objects_pass_through_without_registration():
    predictor = Predictor()
    assert Benchmark.resolve("predict", predictor) is predictor
    assert Benchmark.resolve("interpret", None) is None


def test_unknown_name_lists_the_available_shortcuts():
    with pytest.raises(ValueError, match="Unknown predict 'nope' for Benchmark"):
        Benchmark.resolve("predict", "nope")


def test_stages_run_in_order_and_produce_results(make_benchmark):
    results = make_benchmark().run()
    assert results.evaluations == {"custom": 0.0}
    assert results.interpretations["custom_interpretation"] == {
        "count": 2,
        "metrics": ["custom"],
        "model": "Predictor",
    }
    assert results.task is None


def test_preprocessing_and_embedding_are_fitted_on_the_training_partition_only(make_benchmark):
    benchmark = make_benchmark()
    benchmark.run()
    assert benchmark.components["prepdata"].fitted_on == list(range(8))
    np.testing.assert_array_equal(benchmark.components["embed"].training_targets, np.arange(8) * 2)


@pytest.mark.parametrize("metrics", [["mae", "r2"], ("mae", "r2")])
def test_metrics_accept_lists_and_tuples_keyed_by_the_requested_name(make_benchmark, metrics):
    results = make_benchmark(evaluate=metrics, interpret=None).run()
    assert results.evaluations == {"mae": 0.0, "r2": 1.0}


def test_multiple_predictors_reuse_the_same_prepared_split(make_benchmark):
    first, second = Predictor(), Predictor()
    results = make_benchmark(predict=[first, second], interpret=None).run()
    assert list(results.predictions) == ["Predictor_0", "Predictor_1"]
    assert results.evaluations["Predictor_0"] == {"custom": 0.0}
    assert first.calls == second.calls == 1


def test_splitter_must_provide_a_training_partition(make_benchmark):
    class BadSplitter:
        def split(self, data):
            return {"everything": data}

    with pytest.raises(TypeError, match="must return a mapping containing 'train'"):
        make_benchmark(splitter=BadSplitter()).run()


def test_datasets_of_items_with_targets_are_passed_to_the_predictor_unchanged(make_benchmark):
    class Item:
        def __init__(self, value):
            self.value = value
            self.target = np.asarray([2.0 * value])

    class ItemPrepData:
        def transform(self, data):
            return [Item(value) for value in data]

    class DatasetPredictor:
        def fit(self, data, targets=None):
            self.seen = type(data).__name__

        def predict(self, data):
            return np.asarray([item.target[0] for item in data])

    predictor = DatasetPredictor()
    results = make_benchmark(prepdata=ItemPrepData(), embed=None, predict=predictor, interpret=None).run()
    assert predictor.seen == "list"
    assert results.evaluations == {"custom": 0.0}


def test_two_components_of_one_class_keep_separate_result_keys(make_benchmark):
    class NamedMetric:
        name = "custom"

        def __init__(self, offset):
            self.offset = offset

        def evaluate(self, targets, predictions):
            return self.offset

    results = make_benchmark(evaluate=[NamedMetric(1.0), NamedMetric(2.0)], interpret=None).run()
    assert results.evaluations == {"custom": 1.0, "custom_2": 2.0}


def test_a_component_that_cannot_transform_is_rejected_clearly(make_benchmark):
    class OnlyFitTransform:
        def fit_transform(self, data):
            return data

    with pytest.raises(TypeError, match="must be callable or implement one of"):
        make_benchmark(prepdata=OnlyFitTransform()).run()


def test_records_without_a_representation_are_reported_as_such(make_benchmark):
    import pandas as pd

    class FrameDataset:
        def load(self):
            return pd.DataFrame({"mpids": ["a", "b", "c", "d"], "bg": [1.0, 2.0, 3.0, 4.0]})

    class FrameSplitter:
        def split(self, data):
            return {"train": data.iloc[:2], "test": data.iloc[2:]}

    with pytest.raises(TypeError, match="Prepared data must expose features and targets"):
        make_benchmark(dataset=FrameDataset(), splitter=FrameSplitter(), prepdata=None, embed=None).run()
