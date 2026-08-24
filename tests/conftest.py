"""Shared fixtures for the benchmark tests."""

import pytest

from kalebenchmark import Benchmark
from kalebenchmark.utils.materials.realmat_bag import data_root, require_realmat_bag

from .helpers.fake_components import Dataset, Embed, Interpreter, Metric, Predictor, PrepData, Splitter


@pytest.fixture(scope="session")
def realmat_bag():
    """Skip unless the RealMat-BaG checkout is importable."""
    try:
        return require_realmat_bag()
    except ImportError as error:
        pytest.skip(str(error))


@pytest.fixture(scope="session")
def cif_folder(realmat_bag):
    """Skip unless the CIF files have been extracted."""
    folder = realmat_bag / "cif_file"
    if not (folder / "atom_init.json").is_file():
        pytest.skip(f"CIF files not extracted; run 'unzip cif_file.zip' in {realmat_bag}")
    return folder


@pytest.fixture(scope="session")
def bandgap_root(realmat_bag):
    """Skip unless the band gap JSON files are available."""
    configured = data_root()
    folder = configured if (configured / "data").is_dir() else realmat_bag
    if not (folder / "data" / "fine_tune" / "train_data.json").is_file():
        pytest.skip(f"Band gap data not found under {folder}")
    return folder


@pytest.fixture
def make_benchmark():
    """Return a factory building a fully custom benchmark, with overrides."""

    def factory(**overrides):
        values = dict(
            dataset=Dataset(),
            splitter=Splitter(),
            prepdata=PrepData(),
            embed=Embed(),
            predict=Predictor(),
            evaluate=Metric(),
            interpret=Interpreter(),
        )
        values.update(overrides)
        return Benchmark(**values)

    return factory
