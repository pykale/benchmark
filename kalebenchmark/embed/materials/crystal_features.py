"""Crystal representations for the ``embed`` stage

Aggregates crystal graphs into fixed-size vectors by delegating to the RealMat-BaG featuriser.
No feature aggregation is implemented here.

The stage also accepts a no-op representation, which keeps the crystal graphs as they are for
models that consume graphs directly. That shortcut resolves to
:class:`sklearn.preprocessing.FunctionTransformer`, so no passthrough class is defined here.

This is the benchmark counterpart of :mod:`kale.embed`.
"""

from typing import Any, Optional, Tuple

import numpy as np

from kalebenchmark.utils.materials.realmat_bag import PathLike, require_realmat_bag


class CrystalFeatures:
    """Aggregation of crystal graphs into fixed-size vectors for classical models.

    Args:
        root (str or Path, optional): Checkout root. Defaults to ``REALMAT_BAG_ROOT`` or
            ``./bandgap-benchmark``.
    """

    def __init__(self, root: Optional[PathLike] = None) -> None:
        self.root = root

    def transform(self, data: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Featurise one partition.

        Args:
            data: A crystal graph dataset.

        Returns:
            tuple: The features of shape (n_samples, n_features) and the targets.
        """
        # TODO(pykale): use kale.prepdata.materials_features.extract_features once released.
        require_realmat_bag(self.root)
        from realmat_bag.loaddata.dataloader import extract_features

        return extract_features(data)
