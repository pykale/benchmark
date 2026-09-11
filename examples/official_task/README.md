# Official task

A `TaskCard` fixes the dataset, the split and the metrics of a track; the representation and the
model stay open. Only benchmarks built with `Benchmark.from_task(...)` may be submitted, and
`submit()` sends the predictions the run already produced rather than predicting again.

The example also shows that passing a fixed stage raises immediately:

```text
ValueError: 'splitter' is fixed by TaskCard 'experimental_bg_ood' and cannot be overridden.
```

## Usage

Run from the repository root:

```bash
python -m examples.official_task.main --task experimental_bg_ood --data-root bandgap-benchmark
```
