# Model stage

The `predict` stage accepts any component exposing `fit(X, y)` and `predict(X)`.

The core package defines no estimator or training loop. Reusable implementations should come
from PyKale; experiment-specific adapters belong beside the example that uses them.
