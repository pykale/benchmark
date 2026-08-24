"""Tests for the domain-neutral splitters."""

import numpy as np
import pandas as pd
import pytest

from kalebenchmark.splitdata.dataset_split import PredefinedSplit, RandomSplit


def test_random_split_covers_every_record():
    frame = pd.DataFrame({"id": [f"r{index}" for index in range(10)], "y": np.arange(10.0)})
    partitions = RandomSplit(ratios=(0.6, 0.2, 0.2), seed=0).split(frame)
    assert sorted(partitions) == ["test", "train", "validation"]
    assert sum(len(partition) for partition in partitions.values()) == len(frame)


def test_predefined_split_by_positional_index():
    partitions = PredefinedSplit({"train": [0, 1, 2], "test": [3]}).split(list("abcd"))
    assert partitions == {"train": ["a", "b", "c"], "test": ["d"]}


def test_predefined_split_by_identifier_on_a_frame():
    frame = pd.DataFrame({"accession": ["p1", "p2", "p3"], "y": [1.0, 2.0, 3.0]})
    partitions = PredefinedSplit({"train": ["p1", "p3"], "test": ["p2"]}, key="accession").split(frame)
    assert list(partitions["train"]["accession"]) == ["p1", "p3"]
    assert list(partitions["test"]["accession"]) == ["p2"]


def test_predefined_split_by_identifier_on_records_without_a_frame():
    class Sample:
        def __init__(self, patient_id):
            self.patient_id = patient_id

    data = [Sample("a"), Sample("b"), Sample("c")]
    partitions = PredefinedSplit({"train": ["a", "c"], "test": ["b"]}, key="patient_id").split(data)
    assert [sample.patient_id for sample in partitions["train"]] == ["a", "c"]
    assert [sample.patient_id for sample in partitions["test"]] == ["b"]


def test_predefined_split_reports_members_that_match_nothing():
    frame = pd.DataFrame({"accession": ["p1"], "y": [1.0]})
    with pytest.raises(ValueError, match="Partition 'test' matched none of its"):
        PredefinedSplit({"train": ["p1"], "test": ["absent"]}, key="accession").split(frame)


def test_members_can_be_produced_lazily_by_a_subclass():
    class FoldSplit(PredefinedSplit):
        """An outer fold whose index lists are computed when the split runs."""

        def __init__(self, fold, num_folds):
            super().__init__()
            self.fold, self.num_folds = fold, num_folds
            self.reads = 0

        def members(self, data):
            self.reads += 1
            held_out = list(range(self.fold, 10, self.num_folds))
            return {"train": [index for index in range(10) if index not in held_out], "test": held_out}

    splitter = FoldSplit(fold=1, num_folds=5)
    assert splitter.reads == 0
    partitions = splitter.split(list(range(10)))
    assert splitter.reads == 1
    assert partitions["test"] == [1, 6]
    assert len(partitions["train"]) == 8


def test_random_split_refuses_to_drop_records_the_ratios_do_not_cover():
    with pytest.raises(ValueError, match="leave 10 of 100 records in no partition"):
        RandomSplit(ratios=(0.6, 0.2, 0.1), seed=0).split(list(range(100)))


def test_predefined_split_warns_when_members_are_only_partly_present(caplog):
    frame = pd.DataFrame({"mpids": ["a", "b"], "bg": [1.0, 2.0]})
    splitter = PredefinedSplit({"train": ["a"], "test": ["b", "absent"]}, key="mpids")
    with caplog.at_level("WARNING"):
        partitions = splitter.split(frame)
    assert len(partitions["test"]) == 1
    assert "matched 1 of its 2 members" in caplog.text


def test_json_split_reads_membership_from_files(tmp_path):
    import json

    from kalebenchmark.splitdata.dataset_split import JsonSplit

    (tmp_path / "train.json").write_text(json.dumps({"a": {"y": 1.0}, "c": {"y": 3.0}}))
    (tmp_path / "test.json").write_text(json.dumps(["b"]))
    frame = pd.DataFrame({"accession": ["a", "b", "c"], "y": [1.0, 2.0, 3.0]})

    splitter = JsonSplit({"train": tmp_path / "train.json", "test": tmp_path / "test.json"}, key="accession")
    partitions = splitter.split(frame)
    assert list(partitions["train"]["accession"]) == ["a", "c"]
    assert list(partitions["test"]["accession"]) == ["b"]


def test_a_callable_is_a_valid_splitter_stage(tmp_path):
    """The splitter stage needs no class: any callable taking the records works."""
    import json
    from functools import partial

    from kalebenchmark import Benchmark
    from kalebenchmark.pipeline import _method
    from kalebenchmark.splitdata.dataset_split import JsonSplit

    (tmp_path / "train.json").write_text(json.dumps({"a": {}, "b": {}}))
    (tmp_path / "test.json").write_text(json.dumps({"c": {}}))

    def published_split(data, root):
        return JsonSplit({"train": root / "train.json", "test": root / "test.json"}, key="id").split(data)

    splitter = partial(published_split, root=tmp_path)
    frame = pd.DataFrame({"id": ["a", "b", "c"], "y": [1.0, 2.0, 3.0]})
    partitions = _method(splitter, ("split",), frame)
    assert [len(partition) for partition in partitions.values()] == [2, 1]
    assert Benchmark.resolve("splitter", splitter) is splitter
