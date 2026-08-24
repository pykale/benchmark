"""Type aliases shared across stages"""

import os
from typing import Union

PathLike = Union[str, "os.PathLike[str]"]
