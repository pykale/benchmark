"""Task cards fix a track's protocol and gate leaderboard submission."""

import numpy as np
import pytest

from kalebenchmark import Benchmark, TaskCard

from .helpers.fake_components import Dataset, Embed, Interpreter, Predictor, PrepData, Splitter


@pytest.fixture
def card():
    return TaskCard("track", dataset=Dataset(), splitter=Splitter(), evaluate="mae")


@pytest.mark.parametrize("stage", ["dataset", "splitter", "evaluate"])
def test_fixed_components_cannot_be_overridden(card, stage):
    with pytest.raises(ValueError, match=f"'{stage}' is fixed by TaskCard 'track'"):
        Benchmark.from_task(card, **{stage: Splitter()})


@pytest.mark.parametrize(
    "stage, component",
    [("prepdata", PrepData()), ("embed", Embed()), ("predict", Predictor()), ("interpret", Interpreter())],
)
def test_free_components_can_be_customised(card, stage, component):
    values = dict(prepdata=PrepData(), embed=Embed(), predict=Predictor())
    values[stage] = component
    benchmark = Benchmark.from_task(card, **values)
    assert benchmark.components[stage] is component
    assert benchmark.run().task is card


def test_unknown_task_name_is_rejected():
    with pytest.raises(ValueError, match="Unknown task 'nope'"):
        Benchmark.from_task("nope")


def test_non_task_argument_is_rejected():
    with pytest.raises(TypeError, match="task must be a task name or TaskCard"):
        Benchmark.from_task(object())


def test_exploration_runs_cannot_submit(make_benchmark):
    with pytest.raises(RuntimeError, match="not associated"):
        make_benchmark().run().submit()


def test_official_runs_can_submit(card):
    results = Benchmark.from_task(card, prepdata=PrepData(), embed=Embed(), predict=Predictor()).run()
    payload = results.submit()
    assert payload["task"] == "track"
    assert payload["evaluations"] == {"mae": 0.0}


def test_submit_uses_stored_predictions_without_predicting_again(card):
    predictor = Predictor()
    submitted = []
    results = Benchmark.from_task(
        card,
        prepdata=PrepData(),
        embed=Embed(),
        predict=predictor,
        submitter=lambda payload: submitted.append(payload) or "ok",
    ).run()
    assert predictor.calls == 1
    assert results.submit() == "ok"
    assert predictor.calls == 1
    np.testing.assert_array_equal(submitted[0]["predictions"], results.predictions)
