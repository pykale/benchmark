"""RealMat-BaG: experimental band gap prediction for semiconductor materials

One discipline, one module. Everything materials-specific that a user selects by name is
declared here: the shortcuts, the official tracks, and the defaults. Maintainers of other
disciplines never read this file, and adding a discipline never edits
:mod:`kalebenchmark.benchmark`.

Reference:
    https://github.com/Shef-AIRE/bandgap-benchmark
"""

from typing import Any, Dict

from kalebenchmark.benchmark import Benchmark, GENERIC_BUILTINS, merge_builtins
from kalebenchmark.embed.materials.crystal_features import CrystalFeatures
from kalebenchmark.evaluation.materials.metrics import mean_relative_error
from kalebenchmark.loaddata.materials.bandgap_datasets import experimental_measurements
from kalebenchmark.model.materials.crystal_gnn import CrystalGraphRegressor
from kalebenchmark.prepdata.materials.crystal_graph import CIFPrepData
from kalebenchmark.splitdata.materials.bandgap_split import feature_ood_split
from kalebenchmark.task import TaskCard

MATERIALS_BUILTINS: Dict[str, Dict[str, Any]] = {
    "dataset": {"experimental_bg": experimental_measurements},
    "splitter": {"feature_ood_split": feature_ood_split},
    "prepdata": {"cif": CIFPrepData},
    "embed": {"crystal_features": CrystalFeatures},
    "predict": {"cgcnn": CrystalGraphRegressor},
    "evaluate": {"mrae": mean_relative_error},
}

MATERIALS_TASKS: Dict[str, TaskCard] = {
    "experimental_bg": TaskCard(
        "experimental_bg",
        dataset="experimental_bg",
        splitter="random_split",
        evaluate=["mae", "mrae"],
        description="Experimental band gaps with a random split; representation and model are free.",
    ),
    "experimental_bg_ood": TaskCard(
        "experimental_bg_ood",
        dataset="experimental_bg",
        splitter="feature_ood_split",
        evaluate=["mae", "mrae"],
        description="Experimental band gaps with the official feature out-of-distribution split.",
    ),
}


class RealMatBaG(Benchmark):
    """Band gap benchmark on the RealMat-BaG experimental measurements.

    Adds the materials shortcuts and tracks to the domain-neutral ones, and defaults every stage
    to the standard classical protocol, so that the shortest useful run is ``RealMatBaG().run()``.
    Every default is still replaceable, by name or by object.

    Args:
        dataset: Defaults to the experimental band gap measurements.
        splitter: Defaults to a random split.
        prepdata: Defaults to crystal graphs built from CIF files.
        embed: Defaults to the aggregated crystal features.
        predict: Defaults to support vector regression.
        evaluate: Defaults to mean absolute and mean relative error.
        interpret: Defaults to no interpretation.

    Examples:
        >>> from kalebenchmark.benchmarks.materials.bandgap import RealMatBaG
        >>> RealMatBaG(predict="random_forest").run()  # doctest: +SKIP
        >>> RealMatBaG.from_task("experimental_bg_ood", predict="svr").run()  # doctest: +SKIP
    """

    BUILTINS = merge_builtins(GENERIC_BUILTINS, MATERIALS_BUILTINS)
    TASKS = MATERIALS_TASKS

    def __init__(
        self,
        dataset: Any = "experimental_bg",
        splitter: Any = "random_split",
        prepdata: Any = "cif",
        embed: Any = "crystal_features",
        predict: Any = "svr",
        evaluate: Any = ("mae", "mrae"),
        interpret: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            dataset=dataset,
            splitter=splitter,
            prepdata=prepdata,
            embed=embed,
            predict=predict,
            evaluate=evaluate,
            interpret=interpret,
            **kwargs,
        )
