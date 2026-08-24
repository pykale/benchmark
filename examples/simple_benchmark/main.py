"""The simplest benchmark: every stage selected by name.

Reads left to right as which data, how split, how prepared, how represented,
how predicted and how evaluated.

Usage:
    python examples/simple_benchmark/main.py --data-root bandgap-benchmark
"""

import argparse

from kalebenchmark.benchmarks.materials.bandgap import RealMatBaG


def arg_parse():
    parser = argparse.ArgumentParser(description="Band-gap benchmark with built-in components")
    parser.add_argument("--data-root", default=None, help="RealMat-BaG checkout root", type=str)
    return parser.parse_args()


def main():
    args = arg_parse()
    if args.data_root:
        import os

        os.environ["REALMAT_BAG_ROOT"] = args.data_root

    # ---- compose the benchmark ----
    benchmark = RealMatBaG(
        dataset="experimental_bg",
        splitter="random_split",
        prepdata="cif",
        embed="crystal_features",
        predict="svr",
        evaluate=["mae", "mrae"],
    )

    # ---- run and report ----
    results = benchmark.run()
    for name, value in results.evaluations.items():
        print("{}: {:.4f}".format(name, value))


if __name__ == "__main__":
    main()
