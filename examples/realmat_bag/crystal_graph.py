"""Crystal graph preparation for the ``prepdata`` stage

Turns identifier/target records into crystal graphs by delegating to the RealMat-BaG ``CIFData``
dataset. No neighbour search, Gaussian expansion or atom initialisation is implemented here.

This is the benchmark counterpart of :mod:`kale.prepdata`.
"""

from pathlib import Path
from typing import Any, Optional

import pandas as pd

from examples.realmat_bag.realmat_bag import data_root, PathLike, require_realmat_bag


class CIFPrepData:
    """Preparation of identifier/target records into crystal graphs.

    Args:
        cif_folder (str, optional): Folder of ``<mpid>.cif`` files. Defaults to
            ``<checkout>/cif_file``.
        init_file (str, optional): Atom initialisation JSON file. Defaults to ``atom_init.json``
            inside ``cif_folder``.
        max_nbrs (int, optional): Maximum number of neighbours per atom. Defaults to 12.
        radius (float, optional): Neighbour search radius. Defaults to 8.0.
        root (str or Path, optional): Base directory used only to resolve default data paths.
    """

    def __init__(
        self,
        cif_folder: Optional[PathLike] = None,
        init_file: Optional[PathLike] = None,
        max_nbrs: int = 12,
        radius: float = 8.0,
        root: Optional[PathLike] = None,
    ) -> None:
        self.root = data_root(root)
        self.cif_folder = Path(cif_folder or self.root / "cif_file")
        self.init_file = Path(init_file or self.cif_folder / "atom_init.json")
        self.max_nbrs = max_nbrs
        self.radius = radius

    def transform(self, data: Any) -> Any:
        """Build the crystal graph dataset for one partition.

        Args:
            data: Records whose first two columns are the identifier and the target.

        Returns:
            torch.utils.data.Dataset: A ``CIFData`` dataset.
        """
        # TODO(pykale): use kale.loaddata.materials_datasets.CIFData once released.
        require_realmat_bag(self.root)
        from realmat_bag.loaddata.cifdata import CIFData

        frame = data if hasattr(data, "iloc") else pd.DataFrame(list(data))
        return CIFData(
            frame.iloc[:, :2],
            str(self.cif_folder),
            str(self.init_file),
            self.max_nbrs,
            self.radius,
            False,
        )
