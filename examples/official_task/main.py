"""An official track: the task card fixes the protocol, the user brings a model.

The task card fixes the dataset, the split and the metrics. Everything else is
open, and only benchmarks built through ``from_task`` can be submitted.

Usage:
    python examples/official_task/main.py --data-root bandgap-benchmark
"""

import argparse
import os

from kalebenchmark.benchmarks.materials.bandgap import RealMatBaG


def arg_parse():
    parser = argparse.ArgumentParser(description="Band-gap benchmark on an official track")
    parser.add_argument("--data-root", default=None, help="RealMat-BaG checkout root", type=str)
    parser.add_argument("--task", default="experimental_bg_ood", help="task card name", type=str)
    return parser.parse_args()


def main():
    args = arg_parse()
    if args.data_root:
        os.environ["REALMAT_BAG_ROOT"] = args.data_root

    # ---- the task supplies dataset, splitter and metrics ----
    benchmark = RealMatBaG.from_task(
        args.task,
        prepdata="cif",
        embed="crystal_features",
        predict="svr",
    )
    print("task: {}".format(benchmark.task.description))

    # ---- run, then submit the predictions that were already produced ----
    results = benchmark.run()
    payload = results.submit()
    print("submitted to '{}': {}".format(payload["task"], payload["evaluations"]))

    # ---- a fixed stage cannot be redefined ----
    try:
        RealMatBaG.from_task(args.task, splitter="random_split")
    except ValueError as error:
        print("rejected as expected: {}".format(error))


if __name__ == "__main__":
    main()
