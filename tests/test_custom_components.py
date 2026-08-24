"""Every stage must accept a user implementation without registration."""

import numpy as np

from .helpers.fake_components import Dataset, Embed, Interpreter, Metric, Predictor, PrepData, Splitter


def test_custom_dataset(make_benchmark):
    class MyDataset:
        def load(self):
            return list(range(20, 30))

    results = make_benchmark(dataset=MyDataset()).run()
    assert results.interpretations["custom_interpretation"]["count"] == 2


def test_custom_splitter(make_benchmark):
    class MySplitter:
        def split(self, data):
            return {"train": data[:6], "validation": data[6:]}

    benchmark = make_benchmark(splitter=MySplitter())
    benchmark.run()
    assert benchmark.components["prepdata"].fitted_on == list(range(6))


def test_custom_prepdata(make_benchmark):
    class MyPrepData:
        def fit_transform(self, data):
            return self.transform(data)

        def transform(self, data):
            values = np.asarray(data, dtype=float)
            return values[:, None], values * 2

    results = make_benchmark(prepdata=MyPrepData()).run()
    assert results.evaluations == {"custom": 0.0}


def test_custom_embed(make_benchmark):
    class MyEmbedding:
        def transform(self, data):
            features, targets = data
            return np.hstack([features, features**2]), targets

    class OffsetPredictor:
        def fit(self, x, y):
            self.offset = float(np.mean(y - 2 * x[:, 0]))

        def predict(self, x):
            return 2 * x[:, 0] + self.offset

    results = make_benchmark(embed=MyEmbedding(), predict=OffsetPredictor()).run()
    assert results.evaluations == {"custom": 0.0}


def test_custom_predict(make_benchmark):
    class MyPredictor:
        def fit(self, x, y):
            self.mean = float(np.mean(y))
            return self

        def predict(self, x):
            return np.full(len(x), self.mean)

    results = make_benchmark(predict=MyPredictor()).run()
    assert results.evaluations["custom"] > 0.0


def test_custom_evaluate(make_benchmark):
    def my_metric(y_true, y_pred):
        return float(np.max(np.abs(np.asarray(y_true) - np.asarray(y_pred))))

    results = make_benchmark(evaluate=my_metric, interpret=None).run()
    assert results.evaluations == {"my_metric": 0.0}


def test_custom_interpret(make_benchmark):
    class MyInterpreter:
        name = "targets_seen"

        def interpret(self, targets, representations):
            return {"targets": len(targets), "features": representations.shape[1]}

    results = make_benchmark(interpret=MyInterpreter()).run()
    assert results.interpretations["targets_seen"] == {"targets": 2, "features": 1}


def test_every_stage_replaced_at_once(make_benchmark):
    benchmark = make_benchmark(
        dataset=Dataset(),
        splitter=Splitter(),
        prepdata=PrepData(),
        embed=Embed(),
        predict=Predictor(),
        evaluate=Metric(),
        interpret=Interpreter(),
    )
    results = benchmark.run()
    assert results.evaluations == {"custom": 0.0}
    assert sorted(results.interpretations) == ["custom_interpretation"]
