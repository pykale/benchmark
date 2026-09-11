"""Tests for the RealMat-BaG band gap benchmark.

Everything here needs the RealMat-BaG checkout, and the end-to-end tests also need its extracted
CIF files; both are skipped when missing. No other discipline's tests import this module.
"""

import numpy as np
import pytest

from examples.realmat_bag import RealMatBaG
from examples.realmat_bag.bandgap_datasets import experimental_measurements
from examples.realmat_bag.bandgap_split import feature_ood_split
from examples.realmat_bag.crystal_graph import CIFPrepData


@pytest.fixture
def small_crystal_split(bandgap_root, cif_folder):
    """A 40-material train/test split of real crystals."""

    class Subset:
        def load(self):
            return experimental_measurements(root=bandgap_root).head(40)

    class Splitter:
        def split(self, data):
            return {"train": data.iloc[:30], "test": data.iloc[30:]}

    return Subset(), Splitter(), CIFPrepData(cif_folder=cif_folder)


@pytest.mark.parametrize("stage", sorted(RealMatBaG.BUILTINS))
def test_every_shortcut_resolves(stage):
    for name in RealMatBaG.BUILTINS[stage]:
        assert RealMatBaG.resolve(stage, name) is not None


def test_the_materials_shortcuts_extend_the_generic_ones():
    from kalebenchmark import Benchmark

    assert set(Benchmark.BUILTINS["predict"]) < set(RealMatBaG.BUILTINS["predict"])
    assert "cgcnn" in RealMatBaG.BUILTINS["predict"]
    assert "experimental_bg" in RealMatBaG.BUILTINS["dataset"]


def test_registered_tasks_leave_the_modelling_stages_open():
    for name, task in RealMatBaG.TASKS.items():
        assert task.name == name
        assert task.dataset and task.splitter and task.evaluate
        assert task.prepdata is task.embed is task.predict is task.interpret is None


def test_mrae_shortcut_matches_the_reference_definition(realmat_bag):
    metric = RealMatBaG.resolve("evaluate", "mrae")
    assert metric([2.0, 4.0], [1.0, 2.0]) == pytest.approx(0.5)


def test_experimental_bandgap_loads_the_full_measurement_set(bandgap_root):
    frame = experimental_measurements(root=bandgap_root)
    assert list(frame.columns) == ["mpids", "bg"]
    assert len(frame) == 1705
    assert frame["mpids"].is_unique


def test_feature_ood_split_selects_the_official_partitions(bandgap_root):
    frame = experimental_measurements(root=bandgap_root)
    partitions = feature_ood_split(frame, root=bandgap_root)
    assert len(partitions["train"]) == 1516
    assert len(partitions["test"]) == 189
    assert set(partitions["train"]["mpids"]).isdisjoint(partitions["test"]["mpids"])


@pytest.mark.slow
def test_classical_crystal_pipeline_end_to_end(small_crystal_split):
    dataset, splitter, prepdata = small_crystal_split
    results = RealMatBaG(
        dataset=dataset,
        splitter=splitter,
        prepdata=prepdata,
        embed="crystal_features",
        predict="svr",
        evaluate=["mae", "mrae"],
    ).run()
    assert len(results.predictions) == 10
    assert results.evaluations["mae"] > 0.0


@pytest.mark.slow
def test_crystal_graph_regressor_end_to_end(small_crystal_split):
    dataset, splitter, prepdata = small_crystal_split
    from examples.realmat_bag.crystal_gnn import CrystalGraphRegressor

    results = RealMatBaG(
        dataset=dataset,
        splitter=splitter,
        prepdata=prepdata,
        embed="identity",
        predict=CrystalGraphRegressor(
            model="cgcnn",
            model_params={"atom_fea_len": 16, "n_conv": 1, "h_fea_len": 16},
            max_epochs=1,
            batch_size=8,
        ),
        evaluate="mae",
    ).run()
    assert len(results.predictions) == 10
    assert np.isfinite(results.evaluations["mae"])


@pytest.mark.slow
def test_permutation_importance_interpretation(small_crystal_split):
    dataset, splitter, prepdata = small_crystal_split
    results = RealMatBaG(
        dataset=dataset,
        splitter=splitter,
        prepdata=prepdata,
        embed="crystal_features",
        predict="random_forest",
        evaluate="mae",
        interpret="permutation_importance",
    ).run()
    importances = results.interpretations["permutation_importance"].importances_mean
    assert len(importances) == 225


def test_another_property_flows_through_prepdata(tmp_path, cif_folder):
    """The measured property is an argument, so it must not be hard-coded downstream."""
    import json

    from examples.realmat_bag.crystal_graph import CIFPrepData

    path = tmp_path / "formation.json"
    path.write_text(json.dumps({"mp-571164": {"e_form": -1.2}, "mp-5986": {"e_form": -0.7}}))

    frame = experimental_measurements(target_key="e_form", paths=path)
    assert frame.columns.tolist() == ["mpids", "e_form"]
    dataset = CIFPrepData(cif_folder=cif_folder).transform(frame)
    assert [round(float(dataset[index].target), 1) for index in range(len(dataset))] == [-1.2, -0.7]


def test_other_published_regimes_need_no_materials_code(bandgap_root):
    """A domain split fold is the same protocol with different files."""
    from kalebenchmark.splitdata.dataset_split import JsonSplit

    fold = bandgap_root / "data" / "domain_split" / "chemsys_split"
    if not fold.is_dir():
        pytest.skip(f"domain splits not found under {fold}")
    frame = experimental_measurements(root=bandgap_root)
    partitions = JsonSplit(
        {"train": fold / "mf.chemsys.k0_outer.train.json", "test": fold / "mf.chemsys.k0_outer.test.json"},
        key="mpids",
    ).split(frame)
    assert len(partitions["train"]) == 1526
    assert len(partitions["test"]) == 179


@pytest.mark.slow
def test_a_second_architecture_trains_through_the_same_wrapper(small_crystal_split):
    """Batching and training are architecture independent; only the network differs."""
    from examples.realmat_bag.crystal_gnn import CrystalGraphRegressor

    dataset, splitter, prepdata = small_crystal_split
    results = RealMatBaG(
        dataset=dataset,
        splitter=splitter,
        prepdata=prepdata,
        embed="identity",
        predict=CrystalGraphRegressor(
            model="leftnet",
            model_params={"hidden_channels": 32, "num_layers": 2, "num_radial": 16},
            max_epochs=1,
            batch_size=8,
        ),
        evaluate="mae",
    ).run()
    assert len(results.predictions) == 10
    assert np.isfinite(results.evaluations["mae"])
