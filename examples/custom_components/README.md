# Custom components

Custom and built-in components mix freely. `ElementCountDescriptor`, `MeanPredictor` and
`max_absolute_error` are defined in `main.py`, are not registered anywhere, and require no change
to `kalebenchmark`.

The run also compares two predictors on the same prepared split, so the results are keyed by
model name.

## Usage

```bash
python main.py --data-root ../../bandgap-benchmark
```
