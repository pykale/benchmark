"""JSON record access used by the RealMat-BaG example.

Measurements are often published as ``{identifier: {property: value}}`` JSON. This module reads
that shape into an identifier/target frame, with the identifier field and the property both given
by the caller, so one loader serves any discipline and any property of a dataset.

The stage stays representation independent: it yields identifiers and targets only, leaving the
representation to ``prepdata`` and ``embed``.
"""

import json
from pathlib import Path
from typing import Iterable, List, Sequence, Union

import pandas as pd

from kalebenchmark.utils.typing import PathLike


def load_json_records(path: PathLike, id_column: str = "id", target_key: str = "target") -> pd.DataFrame:
    """Read a ``{identifier: {property: value}}`` file into a two-column frame.

    Args:
        path (str or Path): Path to the JSON file.
        id_column (str, optional): Name to give the identifier column. Defaults to "id".
        target_key (str, optional): Property to read as the target. Defaults to "target".

    Returns:
        pandas.DataFrame: A frame with columns ``[id_column, target_key]``.
    """
    with open(path, encoding="utf-8") as stream:
        records = json.load(stream)
    return pd.DataFrame(
        {id_column: list(records), target_key: [float(record[target_key]) for record in records.values()]}
    )


class JsonRecords:
    """Measurements published as ``{identifier: {property: value}}`` JSON files.

    Args:
        paths (str or Sequence[str]): JSON files to concatenate.
        id_column (str, optional): Name to give the identifier column. Defaults to "id".
        target_key (str, optional): Property to read as the target. Defaults to "target".
    """

    def __init__(
        self,
        paths: Union[str, Path, Sequence[PathLike]],
        id_column: str = "id",
        target_key: str = "target",
    ) -> None:
        selected: Iterable[PathLike] = [paths] if isinstance(paths, (str, Path)) else paths
        self.paths: List[Path] = [Path(path) for path in selected]
        self.id_column = id_column
        self.target_key = target_key

    def load(self) -> pd.DataFrame:
        """Load the measurements.

        Returns:
            pandas.DataFrame: One row per identifier, with identifier and target columns.
        """
        frames = [load_json_records(path, self.id_column, self.target_key) for path in self.paths]
        return pd.concat(frames, ignore_index=True).drop_duplicates(subset=self.id_column, ignore_index=True)
