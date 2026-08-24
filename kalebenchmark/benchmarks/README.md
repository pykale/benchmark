# Concrete benchmarks

One folder per discipline (`materials/` today; `protein/`, `cancer/` and others later), and one
module per benchmark inside it. A module declares everything its users select by name — the
shortcut registry, the official task cards, and the defaults — as a `Benchmark` subclass:

```python
class RealMatBaG(Benchmark):
    BUILTINS = merge_builtins(GENERIC_BUILTINS, MATERIALS_BUILTINS)
    TASKS = MATERIALS_TASKS
```

Adding a discipline means adding a folder here plus a folder in each stage subpackage it needs.
It never means editing `kalebenchmark/benchmark.py`, and a maintainer never has to read another
discipline's code.

## Subclass contract

`Benchmark.from_task` constructs with `cls(...)` and `Benchmark.run` calls `self._execute`, so a
subclass must:

- keep the seven stage keyword arguments, giving any extra argument a default and forwarding
  `**kwargs` to `super().__init__`;
- override `_execute` — not `run` — to change the protocol, for instance cross-validation or a
  pretrain/fine-tune schedule, so that later changes to name resolution and result assembly are
  inherited automatically.

`tests/test_subclass.py` runs a minimal subclass through both paths, so a change that breaks the
contract fails there.
