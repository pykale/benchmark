"""Access to the RealMat-BaG reference checkout

The materials implementations that the benchmark wraps (CIF loading, crystal featurisation,
crystal graph networks and their Lightning trainer) are not part of released PyKale, so they come
from the RealMat-BaG checkout. This module resolves that checkout and makes it importable; every
wrapper that needs it calls :func:`require_realmat_bag` before its function-local import.

Only what released PyKale genuinely provides is imported from ``kale`` — currently
:func:`kale.loaddata.dataset_access.split_by_ratios` and :func:`kale.utils.seed.set_seed`.

TODO(pykale): delete this module once the materials work is released in ``kale.loaddata``,
``kale.prepdata``, ``kale.embed``, ``kale.evaluate`` and ``kale.pipeline``.

Reference:
    https://github.com/Shef-AIRE/bandgap-benchmark
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

from kalebenchmark.utils.typing import PathLike  # re-exported for the materials modules

# Default location of the RealMat-BaG checkout providing the materials data and models.
DEFAULT_DATA_ROOT = "bandgap-benchmark"


def data_root(root: Optional[PathLike] = None) -> Path:
    """Resolve the root of the RealMat-BaG checkout.

    The root is resolved on each call rather than at import time, so that a script may set
    ``REALMAT_BAG_ROOT`` before composing a benchmark. This is the only channel a name shortcut
    has, because the registry constructs components without arguments.

    Args:
        root (str or Path, optional): Explicit root, which takes precedence over the
            ``REALMAT_BAG_ROOT`` environment variable. Defaults to None.

    Returns:
        Path: The checkout root.
    """
    if root is not None:
        return Path(root).expanduser()
    return Path(os.environ.get("REALMAT_BAG_ROOT", DEFAULT_DATA_ROOT)).expanduser()


def _missing_message(missing: Path) -> str:
    """Explain how to obtain the checkout, naming what was looked for."""
    return (
        f"The materials pipelines need the RealMat-BaG data and models, which are not on PyPI "
        f"and were not found at '{missing}'. Get them with:\n"
        "    git clone https://github.com/Shef-AIRE/bandgap-benchmark\n"
        "    cd bandgap-benchmark && unzip cif_file.zip\n"
        "Then run from beside that folder, or set REALMAT_BAG_ROOT to it. Benchmarks that use "
        "no materials component need none of this."
    )


def materials_path(*parts: str, root: Optional[PathLike] = None) -> Path:
    """Resolve a path inside the checkout, explaining how to get it when it is missing.

    The data files are read directly rather than imported, so they never reach
    :func:`require_realmat_bag`; this gives them the same guidance.

    Args:
        *parts (str): Path segments below the checkout root.
        root (str or Path, optional): Checkout root. Defaults to ``REALMAT_BAG_ROOT`` or
            ``./bandgap-benchmark``.

    Returns:
        Path: The resolved path.

    Raises:
        FileNotFoundError: If it does not exist.
    """
    path = data_root(root).joinpath(*parts)
    if not path.exists():
        error_msg = _missing_message(path)
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)
    return path


def require_realmat_bag(root: Optional[PathLike] = None) -> Path:
    """Make the RealMat-BaG reference package importable.

    The checkout ships no installable package, so its root is placed on ``sys.path`` the first
    time a materials component is used.

    Args:
        root (str or Path, optional): Checkout root. Defaults to ``REALMAT_BAG_ROOT`` or
            ``./bandgap-benchmark``.

    Returns:
        Path: The resolved checkout root.

    Raises:
        ImportError: If the checkout cannot be found.
    """
    root = data_root(root)
    if not (root / "realmat_bag").is_dir():
        error_msg = _missing_message(root / "realmat_bag")
        logging.error(error_msg)
        raise ImportError(error_msg)
    resolved = str(root.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
    return root
