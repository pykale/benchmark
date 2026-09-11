# Simple benchmark

The shortest path: every stage is selected by name, and the whole experiment is one
`Benchmark(...).run()` call.

```
dataset=experimental_bg -> splitter=random_split -> prepdata=cif
    -> embed=crystal_features -> predict=svr -> evaluate=[mae, mrae]
```

## Usage

Run from the repository root:

```bash
python -m examples.simple_benchmark.main --data-root bandgap-benchmark
```

`--data-root` is optional when the RealMat-BaG checkout sits at `./bandgap-benchmark` or
`REALMAT_BAG_ROOT` is set. The CIF archive must be extracted first (`unzip cif_file.zip`).
