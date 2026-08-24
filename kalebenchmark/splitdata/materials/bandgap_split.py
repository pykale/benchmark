# =============================================================================
# Author: Haolin Wang, LWang0101@outlook.com
# =============================================================================

"""RealMat-BaG published splits for the ``splitter`` stage

Selecting records from membership lists is generic and lives in
:class:`kalebenchmark.splitdata.dataset_split.JsonSplit`. Nothing here is specific to a split
regime: this module only names where the published files are and that a material is identified
by ``mpids``.
"""

from typing import Any, Dict, Optional

from kalebenchmark.splitdata.dataset_split import JsonSplit
from kalebenchmark.utils.materials.realmat_bag import data_root, PathLike


def feature_ood_split(
    data: Any,
    train_path: Optional[PathLike] = None,
    test_path: Optional[PathLike] = None,
    root: Optional[PathLike] = None,
) -> Dict[str, Any]:
    """Partition records by the published feature out-of-distribution split.

    A splitter stage may be any callable taking the records, so this needs no class of its own.
    The other published regimes are the same protocol with different files, so they need no code
    either: use :class:`~kalebenchmark.splitdata.dataset_split.JsonSplit` with the paths of a
    domain split fold, such as
    ``data/domain_split/chemsys_split/mf.chemsys.k0_outer.{train,test}.json``, or of a
    leave-one-material-out category.

    Args:
        data: The records produced by the dataset stage, carrying an ``mpids`` column.
        train_path (str, optional): JSON file listing the training materials.
        test_path (str, optional): JSON file listing the held-out materials.
        root (str or Path, optional): Checkout root. Defaults to ``REALMAT_BAG_ROOT`` or
            ``./bandgap-benchmark``.

    Returns:
        dict: The ``"train"`` and ``"test"`` partitions.

    Examples:
        >>> from functools import partial
        >>> from kalebenchmark.splitdata.materials.bandgap_split import feature_ood_split
        >>> splitter = partial(feature_ood_split, root="bandgap-benchmark")  # doctest: +SKIP
    """
    split_dir = data_root(root) / "data" / "splits_feature_ood"
    paths = {
        "train": train_path or split_dir / "train_data.json",
        "test": test_path or split_dir / "test_data.json",
    }
    return JsonSplit(paths, key="mpids").split(data)
