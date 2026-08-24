# =============================================================================
# Author: Haolin Wang, LWang0101@outlook.com
# =============================================================================

"""Domain-neutral partitioning for the ``splitter`` stage

Every protocol here is a membership list: which records belong to which partition. That covers a
published train/test list, an out-of-distribution split, one outer fold of a cross-validation,
leave-one-group-out, and a random draw, which differ only in how the lists are produced. A
subclass produces them in :meth:`PredefinedSplit.members`; selecting the records is shared.

Random partitioning delegates to :func:`kale.loaddata.dataset_access.split_by_ratios`; no split
algorithm is implemented here. Nothing in this module is discipline-specific.
"""

import json
import logging
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from kale.loaddata.dataset_access import split_by_ratios
from kale.utils.seed import set_seed
from kalebenchmark.utils.typing import PathLike


def read_json_ids(path: PathLike, key: Optional[str] = None) -> list:
    """Read the identifiers a JSON membership file lists.

    Accepts the two shapes protocols publish: an object keyed by identifier, or an array of
    identifiers or of records carrying one.

    Args:
        path (str or Path): The JSON file.
        key (str, optional): Field holding the identifier, when the file lists records.

    Returns:
        list: The identifiers, in file order.
    """
    with open(path, encoding="utf-8") as stream:
        records = json.load(stream)
    if isinstance(records, dict):
        return list(records)
    return [record[key] if key and isinstance(record, dict) else record for record in records]


def _take(data: Any, indices: Sequence[int]) -> Any:
    """Select rows of a data frame, or items of a sequence, by position."""
    if hasattr(data, "iloc"):
        return data.iloc[list(indices)].reset_index(drop=True)
    return [data[index] for index in indices]


def _identifier(record: Any, key: str) -> Any:
    """Read the identifying field of one record, whether mapping-like or object-like."""
    try:
        return record[key]
    except (TypeError, KeyError, IndexError):
        return getattr(record, key)


class PredefinedSplit:
    """Partitioning by membership lists.

    Members are positional indices, or values of an identifying field when ``key`` is given. The
    lists come from the protocol and the records from the ``dataset`` stage, so a split never
    re-reads the data.

    Args:
        partitions (Mapping[str, Iterable], optional): Partition name to the members it contains,
            for example ``{"train": [...], "test": [...]}``. A subclass may instead override
            :meth:`members` to produce them.
        key (str, optional): Field identifying a record, such as an accession or a patient
            identifier. Defaults to None, meaning members are positional indices.

    Examples:
        >>> from kalebenchmark.splitdata.dataset_split import PredefinedSplit
        >>> PredefinedSplit({"train": [0, 1, 2], "test": [3]}).split(list("abcd"))
        {'train': ['a', 'b', 'c'], 'test': ['d']}
    """

    def __init__(self, partitions: Optional[Mapping[str, Iterable[Any]]] = None, key: Optional[str] = None) -> None:
        self.partitions = partitions
        self.key = key

    def members(self, data: Any) -> Mapping[str, Iterable[Any]]:
        """Return the members of each partition.

        Override this to produce the lists, whether by reading where a protocol publishes them or
        by deriving them from ``data``. Producing them here rather than in ``__init__`` keeps a
        splitter constructible without its data.

        Args:
            data: The records being split, for splitters that derive membership from them.

        Returns:
            Mapping[str, Iterable]: Partition name to its members.

        Raises:
            ValueError: If no membership lists were given.
        """
        if self.partitions is None:
            raise ValueError(f"{type(self).__name__} needs partitions, or a members() override producing them.")
        return self.partitions

    def split(self, data: Any) -> Dict[str, Any]:
        """Select each partition from the given records.

        Args:
            data: The records produced by the dataset stage.

        Returns:
            dict: One partition per name in :meth:`members`.

        Raises:
            ValueError: If a non-empty partition selects no record, which means the members do
                not match this data.
        """
        partitions = {}
        for name, members in self.members(data).items():
            members = list(members)
            selected = self._select(data, members)
            if members and not len(selected):
                on_key = f" on key '{self.key}'" if self.key else ""
                raise ValueError(f"Partition '{name}' matched none of its {len(members)} members{on_key}.")
            if len(selected) < len(members):
                logging.warning(
                    "Partition '%s' matched %d of its %d members; the rest are absent from the data.",
                    name,
                    len(selected),
                    len(members),
                )
            partitions[name] = selected
        return partitions

    def _select(self, data: Any, members: Sequence[Any]) -> Any:
        """Select the records belonging to one partition."""
        if self.key is None:
            return _take(data, members)
        wanted = set(members)
        if hasattr(data, "iloc"):
            return data[data[self.key].isin(wanted)].reset_index(drop=True)
        return [record for record in data if _identifier(record, self.key) in wanted]


class RandomSplit(PredefinedSplit):
    """Random partitioning, delegating to :func:`kale.loaddata.dataset_access.split_by_ratios`.

    The drawn indices are the membership lists, so selection is inherited.

    Args:
        ratios (Sequence[float], optional): Ratios for the train, validation and test
            partitions. Defaults to (0.7, 0.15, 0.15).
        seed (int, optional): Seed passed to :func:`kale.utils.seed.set_seed`. ``split_by_ratios``
            takes no generator argument and draws from the global torch generator, so seeding is
            how a run is made reproducible. Defaults to 42.
    """

    def __init__(self, ratios: Sequence[float] = (0.7, 0.15, 0.15), seed: Optional[int] = 42) -> None:
        super().__init__()
        self.ratios = list(ratios)
        self.seed = seed

    def members(self, data: Any) -> Mapping[str, Iterable[int]]:
        """Draw the indices of each partition.

        Args:
            data: The records being split; only their number is used.

        Returns:
            Mapping[str, Iterable[int]]: Partition name to its positional indices.

        Raises:
            ValueError: If the ratios leave records in no partition.
        """
        if self.seed is not None:
            set_seed(self.seed)
        names = ["train", "validation", "test"]
        subsets = split_by_ratios(range(len(data)), self.ratios)
        # split_by_ratios appends a leftover when the ratios sum to less than one. An empty
        # leftover is float noise; a real one would be dropped silently by the zip below.
        dropped = sum(len(subset) for subset in subsets[len(names) :])
        if dropped:
            raise ValueError(
                f"ratios={tuple(self.ratios)} leave {dropped} of {len(data)} records in no partition. "
                "Give three ratios that sum to one, or fewer ratios so the remainder becomes the last partition."
            )
        return {name: list(subset.indices) for name, subset in zip(names, subsets)}


class JsonSplit(PredefinedSplit):
    """Membership lists published as one JSON file per partition.

    Args:
        paths (Mapping[str, str]): Partition name to the JSON file listing its members.
        key (str, optional): Field identifying a record, in the data and in the files.
    """

    def __init__(self, paths: Mapping[str, PathLike], key: Optional[str] = None) -> None:
        super().__init__(key=key)
        self.paths = dict(paths)

    def members(self, data: Any) -> Mapping[str, Iterable[Any]]:
        """Read each partition's identifiers, deferred so construction needs no files.

        Args:
            data: The records being split, which these lists do not depend on.

        Returns:
            Mapping[str, Iterable]: Partition name to its identifiers.
        """
        return {name: read_json_ids(path, self.key) for name, path in self.paths.items()}
