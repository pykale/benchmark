"""The subclass contract that concrete benchmarks depend on.

A discipline is added as a ``Benchmark`` subclass carrying its own shortcuts and task cards.
These tests pin the two things such a subclass relies on: ``from_task`` constructs with
``cls(...)``, and ``run`` delegates execution to ``_execute``. A change to the base class that
breaks either fails here rather than in a discipline's own tests.
"""

import pytest

from kalebenchmark import Benchmark, TaskCard
from kalebenchmark.benchmark import GENERIC_BUILTINS, merge_builtins

from .helpers.fake_components import Dataset, Embed, Interpreter, Metric, Predictor, PrepData, Splitter


class DomainBenchmark(Benchmark):
    """A minimal discipline: its own shortcuts, tasks and defaults."""

    BUILTINS = merge_builtins(GENERIC_BUILTINS, {"dataset": {"toy": Dataset}, "splitter": {"toy_split": Splitter}})
    TASKS = {"toy_task": TaskCard("toy_task", dataset="toy", splitter="toy_split", evaluate=Metric())}

    def __init__(self, dataset="toy", splitter="toy_split", prepdata=None, embed=None, **kwargs):
        super().__init__(dataset=dataset, splitter=splitter, prepdata=prepdata, embed=embed, **kwargs)


def test_subclass_shortcuts_extend_rather_than_replace_the_generic_ones():
    assert isinstance(DomainBenchmark.resolve("dataset", "toy"), Dataset)
    assert DomainBenchmark.resolve("splitter", "random_split") is not None


def test_base_class_does_not_know_the_discipline():
    with pytest.raises(ValueError, match="Unknown dataset 'toy' for Benchmark"):
        Benchmark.resolve("dataset", "toy")


def test_subclass_defaults_run_end_to_end():
    results = DomainBenchmark(prepdata=PrepData(), embed=Embed(), predict=Predictor(), evaluate=Metric()).run()
    assert results.evaluations == {"custom": 0.0}


def test_from_task_constructs_the_subclass():
    benchmark = DomainBenchmark.from_task(
        "toy_task", prepdata=PrepData(), embed=Embed(), predict=Predictor(), interpret=Interpreter()
    )
    assert isinstance(benchmark, DomainBenchmark)
    results = benchmark.run()
    assert results.task is DomainBenchmark.TASKS["toy_task"]
    assert results.submit()["task"] == "toy_task"


def test_unknown_task_names_the_subclass():
    with pytest.raises(ValueError, match="Unknown task 'nope' for DomainBenchmark"):
        DomainBenchmark.from_task("nope")


def test_execute_is_the_extension_point_for_a_different_protocol():
    class TwiceBenchmark(DomainBenchmark):
        """A protocol that runs the pipeline twice and reports the second run."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.runs = 0

        def _execute(self, resolved):
            self.runs += 1
            super()._execute(resolved)
            self.runs += 1
            return super()._execute(resolved)

    benchmark = TwiceBenchmark(prepdata=PrepData(), embed=Embed(), predict=Predictor(), evaluate=Metric())
    results = benchmark.run()
    assert benchmark.runs == 2
    assert results.evaluations == {"custom": 0.0}
