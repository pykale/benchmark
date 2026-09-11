"""The same protocol with a graph neural network instead of a classical model.

Only two stages change relative to ``examples/simple_benchmark``: the
representation stays as crystal graphs (``embed="identity"``) and the predictor
is CGCNN, trained by the RealMat-BaG Lightning trainer.

Usage:
    python -m examples.crystal_gnn.main --max-epochs 2 --limit 200
"""

import argparse
import os

from examples.realmat_bag import RealMatBaG
from examples.realmat_bag.bandgap_datasets import experimental_measurements
from examples.realmat_bag.crystal_gnn import CrystalGraphRegressor


class BandgapSubset:
    """The band-gap dataset, optionally truncated to keep the demo short."""

    def __init__(self, limit=None):
        self.limit = limit

    def load(self):
        frame = experimental_measurements()
        return frame.head(self.limit) if self.limit else frame


def arg_parse():
    parser = argparse.ArgumentParser(description="Band-gap benchmark with a crystal graph network")
    parser.add_argument("--data-root", default=None, help="RealMat-BaG checkout root", type=str)
    parser.add_argument("--max-epochs", default=2, help="training epochs", type=int)
    parser.add_argument("--limit", default=200, help="number of materials, 0 for all", type=int)
    return parser.parse_args()


def main():
    args = arg_parse()
    if args.data_root:
        os.environ["REALMAT_BAG_ROOT"] = args.data_root

    # ---- crystal graphs go straight to the network ----
    benchmark = RealMatBaG(
        dataset=BandgapSubset(args.limit or None),
        splitter="random_split",
        prepdata="cif",
        embed="identity",
        predict=CrystalGraphRegressor(max_epochs=args.max_epochs, batch_size=32),
        evaluate=["mae", "mrae"],
    )

    # ---- run and report ----
    results = benchmark.run()
    for name, value in results.evaluations.items():
        print("{}: {:.4f}".format(name, value))


if __name__ == "__main__":
    main()
