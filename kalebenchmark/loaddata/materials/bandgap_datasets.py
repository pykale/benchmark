# =============================================================================
# Author: Haolin Wang, LWang0101@outlook.com
# =============================================================================

"""RealMat-BaG measurements for the ``dataset`` stage

Reading the files is generic and lives in :class:`kalebenchmark.loaddata.json_datasets.JsonRecords`,
where the identifier field and the property are both arguments. Nothing here is specific to a
property: this module only names where the published files are and that a material is identified
by ``mpids``.
"""

from typing import Optional, Sequence, Union

import pandas as pd

from kalebenchmark.loaddata.json_datasets import JsonRecords
from kalebenchmark.utils.materials.realmat_bag import data_root, PathLike


def experimental_measurements(
    target_key: str = "bg",
    paths: Optional[Union[str, Sequence[PathLike]]] = None,
    root: Optional[PathLike] = None,
) -> pd.DataFrame:
    """Load the RealMat-BaG experimental measurements.

    A dataset stage may be any callable returning the records, so this needs no class of its own:
    it supplies the published paths and the ``mpids`` identifier, and hands the rest to
    :class:`~kalebenchmark.loaddata.json_datasets.JsonRecords`.

    Args:
        target_key (str, optional): Property to read as the target. Defaults to "bg", the only
            property the published files carry today.
        paths (str or Sequence[str], optional): JSON files to concatenate. Defaults to the union
            of the official fine-tuning train and test files, so that the splitter owns every
            partitioning decision.
        root (str or Path, optional): Checkout root. Defaults to ``REALMAT_BAG_ROOT`` or
            ``./bandgap-benchmark``.

    Returns:
        pandas.DataFrame: One row per material, with ``mpids`` and target columns.

    Examples:
        >>> from kalebenchmark.loaddata.materials.bandgap_datasets import experimental_measurements
        >>> experimental_measurements().columns.tolist()  # doctest: +SKIP
        ['mpids', 'bg']
    """
    fine_tune = data_root(root) / "data" / "fine_tune"
    records = JsonRecords(
        paths or [fine_tune / "train_data.json", fine_tune / "test_data.json"],
        id_column="mpids",
        target_key=target_key,
    )
    return records.load()
