# Crystal graph network

The same protocol as `examples/simple_benchmark`, with a graph neural network in place of the
classical model. Only the last two stages differ: the crystal graphs are passed through
unchanged (`embed="identity"`) and CGCNN is trained through the RealMat-BaG Lightning trainer.

## Usage

```bash
python main.py --max-epochs 2 --limit 200
```

The defaults are sized as a smoke test. Accuracy at two epochs on 200 materials is not
meaningful; raise `--max-epochs` and set `--limit 0` for a real run.
