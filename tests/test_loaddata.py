"""Tests for the domain-neutral record access."""

import json

import pandas as pd
import pytest

from kalebenchmark.loaddata.json_datasets import JsonRecords, load_json_records


@pytest.fixture
def records(tmp_path):
    """Two files of ``{identifier: {property: value}}`` records, overlapping by one."""
    (tmp_path / "a.json").write_text(json.dumps({"x1": {"y": 1.0, "z": 9.0}, "x2": {"y": 2.0, "z": 8.0}}))
    (tmp_path / "b.json").write_text(json.dumps({"x2": {"y": 2.0, "z": 8.0}, "x3": {"y": 3.0, "z": 7.0}}))
    return tmp_path


def test_the_property_and_the_identifier_are_both_arguments(records):
    frame = load_json_records(records / "a.json", id_column="accession", target_key="z")
    assert frame.columns.tolist() == ["accession", "z"]
    assert frame["z"].tolist() == [9.0, 8.0]


def test_files_are_concatenated_and_deduplicated(records):
    frame = JsonRecords([records / "a.json", records / "b.json"], id_column="accession", target_key="y").load()
    assert frame["accession"].tolist() == ["x1", "x2", "x3"]
    assert frame["accession"].is_unique


def test_a_single_path_needs_no_sequence(records):
    frame = JsonRecords(records / "a.json", id_column="accession", target_key="y").load()
    assert len(frame) == 2


def test_a_missing_property_is_reported(records):
    with pytest.raises(KeyError):
        load_json_records(records / "a.json", target_key="absent")


def test_targets_are_numeric(records):
    frame = JsonRecords(records / "a.json", target_key="y").load()
    assert pd.api.types.is_float_dtype(frame["y"])


def test_a_callable_is_a_valid_dataset_stage(records):
    """The dataset stage needs no class: any zero-argument callable returning records works."""
    from functools import partial

    from kalebenchmark import Benchmark
    from kalebenchmark.pipeline import _method

    load_z = partial(JsonRecords(records / "a.json", id_column="accession", target_key="z").load)
    assert _method(load_z, ("load", "get_data")).columns.tolist() == ["accession", "z"]
    assert Benchmark.resolve("dataset", load_z) is load_z
