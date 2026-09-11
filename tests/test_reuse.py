"""Guard rails against re-implementing what PyKale and RealMat-BaG provide.

Each test patches the existing implementation and asserts that the benchmark calls it, so a
future duplicate implementation fails here rather than silently diverging from the reference
behaviour.
"""

import re
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import kale.loaddata.dataset_access
import kale.utils.seed
import kalebenchmark
from examples.realmat_bag import crystal_features, crystal_graph, metrics
from kalebenchmark.splitdata import dataset_split

PACKAGE = Path(kalebenchmark.__file__).parent


def test_random_split_delegates_to_pykale():
    frame = pd.DataFrame({"mpids": [f"mp-{index}" for index in range(10)], "bg": np.arange(10.0)})
    with patch.object(
        dataset_split, "split_by_ratios", wraps=kale.loaddata.dataset_access.split_by_ratios
    ) as split_by_ratios:
        partitions = dataset_split.RandomSplit(ratios=(0.6, 0.2, 0.2), seed=0).split(frame)
    split_by_ratios.assert_called_once()
    assert sorted(partitions) == ["test", "train", "validation"]
    assert sum(len(partition) for partition in partitions.values()) == len(frame)


def test_random_split_seeds_through_pykale():
    frame = pd.DataFrame({"mpids": ["mp-1", "mp-2", "mp-3", "mp-4"], "bg": [1.0, 2.0, 3.0, 4.0]})
    with patch.object(dataset_split, "set_seed", wraps=kale.utils.seed.set_seed) as set_seed:
        dataset_split.RandomSplit(ratios=(0.5, 0.25, 0.25), seed=7).split(frame)
    set_seed.assert_called_once_with(7)


def test_random_split_is_reproducible_for_a_fixed_seed():
    frame = pd.DataFrame({"mpids": [f"mp-{index}" for index in range(20)], "bg": np.arange(20.0)})
    splitter = dataset_split.RandomSplit(seed=3)
    first, second = splitter.split(frame), splitter.split(frame)
    pd.testing.assert_frame_equal(first["train"], second["train"])


def test_crystal_features_delegates_to_the_reference_featuriser(realmat_bag):
    import realmat_bag.loaddata.dataloader as dataloader

    sentinel = (np.zeros((2, 3)), np.zeros(2))
    with patch.object(dataloader, "extract_features", return_value=sentinel) as extract_features:
        features = crystal_features.CrystalFeatures().transform("dataset")
    extract_features.assert_called_once_with("dataset")
    assert features is sentinel


def test_mean_relative_error_delegates_to_the_reference_metric(realmat_bag):
    import realmat_bag.pipeline.models.classical_ml as classical_ml

    expected = classical_ml.mean_relative_error(np.asarray([1.0, 2.0]), np.asarray([1.5, 2.5]))
    with patch.object(classical_ml, "mean_relative_error", wraps=classical_ml.mean_relative_error) as metric:
        value = metrics.mean_relative_error([1.0, 2.0], [1.5, 2.5])
    metric.assert_called_once()
    assert value == pytest.approx(expected)


def test_cif_prepdata_delegates_to_the_reference_dataset(realmat_bag):
    import realmat_bag.loaddata.cifdata as cifdata

    frame = pd.DataFrame({"mpids": ["mp-1"], "bg": [1.0]})
    with patch.object(cifdata, "CIFData", return_value="dataset") as cif_data:
        prepared = crystal_graph.CIFPrepData(cif_folder="cifs", init_file="cifs/atom_init.json").transform(frame)
    cif_data.assert_called_once()
    assert prepared == "dataset"


@pytest.mark.parametrize(
    "pattern",
    [
        r"train_test_split",
        r"np\.random\.permutation",
        r"torch\.randperm",
        r"def (mean_absolute_error|r2_score|mean_squared_error)\b",
    ],
)
def test_package_does_not_reimplement_existing_functionality(pattern):
    offenders = [
        str(path.relative_to(PACKAGE))
        for path in sorted(PACKAGE.rglob("*.py"))
        if re.search(pattern, path.read_text(encoding="utf-8"))
    ]
    assert offenders == [], f"{pattern} should be delegated, not implemented: {offenders}"
