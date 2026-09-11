# Dataset stage

The `dataset` stage accepts an object exposing `load()` or a callable returning records.

Concrete readers, schemas and data locations belong to examples. This keeps datasets independent
from the domain-neutral pipeline and avoids adding pandas or storage dependencies to the package.
