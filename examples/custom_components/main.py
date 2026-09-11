"""Mixing built-in shortcuts with custom components.

The custom descriptor, predictor and metric are ordinary objects. They are not
registered anywhere and no benchmark source file has to change for them to run.

Usage:
    python -m examples.custom_components.main --data-root bandgap-benchmark
"""

import argparse
import os

import numpy as np

from examples.realmat_bag import RealMatBaG


class ElementCountDescriptor:
    """A deliberately naive representation: crystal size and mean atomic number.

    Implements the ``prepdata`` protocol (``fit`` on train only, then
    ``transform`` per partition) and emits ``(features, targets)``.
    """

    def fit(self, data):
        return self

    def transform(self, data):
        features, targets = [], []
        for item in data:
            atomic_numbers = item.atom_num.numpy()
            features.append([len(atomic_numbers), float(np.mean(atomic_numbers))])
            targets.append(float(item.target))
        return np.asarray(features), np.asarray(targets)


class MeanPredictor:
    """Predicts the training mean; a floor for any real model."""

    def fit(self, x, y):
        self.value = float(np.mean(y))
        return self

    def predict(self, x):
        return np.full(len(x), self.value)


def max_absolute_error(y_true, y_pred):
    """A custom metric supplied as a plain function."""
    return float(np.max(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def arg_parse():
    parser = argparse.ArgumentParser(description="Band-gap benchmark with custom components")
    parser.add_argument("--data-root", default=None, help="RealMat-BaG checkout root", type=str)
    return parser.parse_args()


def main():
    args = arg_parse()
    if args.data_root:
        os.environ["REALMAT_BAG_ROOT"] = args.data_root

    # ---- built-in data and protocol, custom representation and model ----
    benchmark = RealMatBaG(
        dataset="experimental_bg",
        splitter="random_split",
        prepdata="cif",
        embed=ElementCountDescriptor(),
        predict=["svr", MeanPredictor()],
        evaluate=["mae", max_absolute_error],
    )

    # ---- run and report ----
    results = benchmark.run()
    for model, evaluations in results.evaluations.items():
        scores = ", ".join("{}={:.4f}".format(name, value) for name, value in evaluations.items())
        print("{}: {}".format(model, scores))


if __name__ == "__main__":
    main()
