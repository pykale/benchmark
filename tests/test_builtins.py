"""Tests for the dependency-light shortcuts carried by the base benchmark."""

from pathlib import Path

from kalebenchmark import Benchmark
from kalebenchmark.splitdata.dataset_split import PredefinedSplit, RandomSplit


def test_base_benchmark_registers_domain_neutral_shortcuts():
    assert Benchmark.BUILTINS["splitter"] == {
        "random_split": RandomSplit,
        "predefined_split": PredefinedSplit,
    }
    for stage in ("dataset", "prepdata", "embed", "predict", "evaluate", "interpret"):
        assert Benchmark.BUILTINS[stage] == {}


def test_random_split_shortcut_resolves():
    assert isinstance(Benchmark.resolve("splitter", "random_split"), RandomSplit)


def test_predefined_split_shortcut_resolves():
    assert isinstance(Benchmark.resolve("splitter", "predefined_split"), PredefinedSplit)


def test_core_package_has_no_direct_third_party_import_except_kale():
    package = Path(__file__).parents[1] / "kalebenchmark"
    forbidden = ("numpy", "pandas", "sklearn", "shap", "torch", "pytorch_lightning", "realmat_bag")
    offenders = []
    for path in package.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for dependency in forbidden:
            if f"import {dependency}" in text or f"from {dependency}" in text:
                offenders.append(f"{path.relative_to(package)}: {dependency}")
    assert offenders == []
