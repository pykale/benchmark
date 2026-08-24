# =============================================================================
# Author: Haolin Wang, LWang0101@outlook.com
# =============================================================================

"""Benchmark composition (user-facing API)

:class:`Benchmark` stores one component per pipeline stage, resolves built-in name shortcuts,
and delegates execution to :mod:`kalebenchmark.pipeline`. It contains no machine learning logic
of its own, and knows no discipline: its registry holds only domain-neutral shortcuts.

A concrete benchmark is a subclass carrying its own :attr:`Benchmark.BUILTINS` and
:attr:`Benchmark.TASKS`, so that adding a discipline never means editing this module. See
:mod:`kalebenchmark.benchmarks` for the concrete benchmarks and for the subclass contract.
"""

import inspect
import logging
from typing import Any, Dict, Mapping, Optional

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVR

from .interpretation.feature_attribution import PermutationImportance, ShapInterpreter
from .pipeline import _name
from .pipeline import run as run_pipeline
from .pipeline import unique_label
from .results import Results
from .splitdata.dataset_split import RandomSplit
from .task import TaskCard

STAGES = ("dataset", "splitter", "prepdata", "embed", "predict", "evaluate", "interpret")

# Stages where several components may be compared or reported in one run.
MULTI_STAGES = ("predict", "evaluate", "interpret")

# Shortcuts that belong to no discipline. A subclass adds its own on top; see
# kalebenchmark/benchmarks/materials/bandgap.py for an example.
GENERIC_BUILTINS: Dict[str, Dict[str, Any]] = {
    "dataset": {},
    "splitter": {"random_split": RandomSplit},
    "prepdata": {},
    "embed": {"identity": FunctionTransformer},
    "predict": {
        "svr": SVR,
        "random_forest": RandomForestRegressor,
        "linear_regression": LinearRegression,
    },
    "evaluate": {
        "mae": mean_absolute_error,
        "mse": mean_squared_error,
        "r2": r2_score,
    },
    "interpret": {"shap": ShapInterpreter, "permutation_importance": PermutationImportance},
}


def merge_builtins(*registries: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Merge shortcut registries stage by stage, later ones winning.

    A subclass uses this to add its discipline's shortcuts on top of the inherited ones.

    Args:
        *registries: Registries mapping a stage to its ``{name: component}`` shortcuts.

    Returns:
        dict: One merged registry, with an entry for every stage in :data:`STAGES`.
    """
    merged: Dict[str, Dict[str, Any]] = {stage: {} for stage in STAGES}
    for registry in registries:
        for stage, shortcuts in registry.items():
            merged.setdefault(stage, {}).update(shortcuts)
    return merged


class Benchmark:
    """Compose and run replaceable benchmark components in pipeline order.

    Each stage accepts a built-in name or any object providing the expected methods, so a custom
    implementation never has to be registered first.

    This base class carries only domain-neutral shortcuts. Discipline-specific ones live on
    subclasses, which is also where task cards are declared.

    Subclass contract, which :meth:`from_task` and :meth:`run` depend on:

    * keep the seven stage keyword arguments, because ``from_task`` constructs with ``cls(...)``;
      a subclass that adds arguments should give them defaults and forward ``**kwargs``;
    * override :meth:`_execute` to change how a run is executed, rather than :meth:`run`, so that
      later changes to resolution and result assembly are inherited;
    * set :attr:`BUILTINS` with :func:`merge_builtins` and :attr:`TASKS` with the tracks offered.

    Attributes:
        BUILTINS (dict): Name shortcuts per stage. These are conveniences, not a permission list:
            any object with the right methods can be passed to any stage without being registered.
        TASKS (dict): Task cards offered by this benchmark, keyed by name.

    Args:
        dataset: Component providing ``load()``, or a built-in name.
        splitter: Component providing ``split(data)``, or a built-in name.
        prepdata: Optional ``fit``/``transform`` component, or a built-in name.
        embed: Optional ``fit``/``transform`` component, or a built-in name.
        predict: Predictor(s) providing ``fit``/``predict``, or built-in names.
        evaluate: Metric(s) called as ``(y_true, y_pred)``, or built-in names.
        interpret: Optional interpreter(s), or built-in names.

    Examples:
        >>> from kalebenchmark.benchmarks.materials.bandgap import RealMatBaG
        >>> results = RealMatBaG(
        ...     dataset="experimental_bg",
        ...     splitter="random_split",
        ...     prepdata="cif",
        ...     embed="crystal_features",
        ...     predict="svr",
        ...     evaluate=["mae", "mrae"],
        ... ).run()  # doctest: +SKIP
    """

    BUILTINS: Dict[str, Dict[str, Any]] = GENERIC_BUILTINS
    TASKS: Dict[str, TaskCard] = {}

    def __init__(
        self,
        dataset: Any,
        splitter: Any,
        prepdata: Any = None,
        embed: Any = None,
        predict: Any = None,
        evaluate: Any = None,
        interpret: Any = None,
    ) -> None:
        self.components: Dict[str, Any] = {
            "dataset": dataset,
            "splitter": splitter,
            "prepdata": prepdata,
            "embed": embed,
            "predict": predict,
            "evaluate": evaluate,
            "interpret": interpret,
        }
        self.task: Optional[TaskCard] = None
        self.submitter: Optional[Any] = None

    @classmethod
    def resolve(cls, stage: str, component: Any) -> Any:
        """Resolve a built-in name shortcut; return other objects unchanged.

        A registered class is instantiated with its own defaults, while a registered function is
        used as it is. This is what lets ``"svr"`` map straight to :class:`sklearn.svm.SVR` and
        ``"mae"`` to :func:`sklearn.metrics.mean_absolute_error` without a wrapper.

        Args:
            stage (str): One of :data:`STAGES`.
            component: A built-in name, or any user-provided object.

        Returns:
            The resolved component.

        Raises:
            ValueError: If ``component`` is a name that this benchmark does not provide.
        """
        if component is None or not isinstance(component, str):
            return component
        try:
            builtin = cls.BUILTINS[stage][component]
        except KeyError:
            available = ", ".join(sorted(cls.BUILTINS.get(stage, {}))) or "none"
            error_msg = (
                f"Unknown {stage} {component!r} for {cls.__name__}. Available: {available}. "
                "Discipline-specific shortcuts live on the matching Benchmark subclass."
            )
            logging.error(error_msg)
            raise ValueError(error_msg) from None
        return builtin() if inspect.isclass(builtin) else builtin

    @classmethod
    def _resolve_stage(cls, stage: str, component: Any) -> Any:
        """Resolve one stage, keeping the requested names as result labels."""
        if stage not in MULTI_STAGES:
            return cls.resolve(stage, component)
        if isinstance(component, dict):
            return {label: cls.resolve(stage, item) for label, item in component.items()}
        if not isinstance(component, (list, tuple)):
            component = [component]
        labelled: Dict[str, Any] = {}
        for index, item in enumerate(component):
            if item is None:
                continue
            resolved = cls.resolve(stage, item)
            label = item if isinstance(item, str) else _name(resolved, index)
            labelled[unique_label(label, labelled)] = resolved
        return labelled

    @classmethod
    def from_task(cls, task: Any, submitter: Optional[Any] = None, **components: Any) -> "Benchmark":
        """Create a leaderboard-eligible benchmark from a task card.

        The construction path determines eligibility: only benchmarks created here may be
        submitted. Stages that the card fixes cannot be passed.

        Args:
            task (str or TaskCard): A name in :attr:`TASKS`, or a task card.
            submitter (callable, optional): Receives the submission payload.
            **components: Values for the stages the card leaves open.

        Returns:
            Benchmark: A benchmark bound to ``task``.

        Raises:
            TypeError: If ``task`` is neither a known name nor a task card.
            ValueError: If the name is unknown, or a stage fixed by the card is passed.
        """
        task = cls._task_card(task)
        fixed = {name: value for name, value in task.components().items() if value is not None}
        overridden = sorted(fixed.keys() & components.keys())
        if overridden:
            error_msg = f"'{overridden[0]}' is fixed by TaskCard '{task.name}' and cannot be overridden."
            logging.error(error_msg)
            raise ValueError(error_msg)

        benchmark = cls(**{**components, **fixed})
        benchmark.task = task
        benchmark.submitter = submitter
        return benchmark

    @classmethod
    def _task_card(cls, task: Any) -> TaskCard:
        """Resolve a task name against :attr:`TASKS`; pass a task card through."""
        if isinstance(task, TaskCard):
            return task
        if not isinstance(task, str):
            raise TypeError("task must be a task name or TaskCard")
        try:
            return cls.TASKS[task]
        except KeyError:
            available = ", ".join(sorted(cls.TASKS)) or "none"
            error_msg = f"Unknown task {task!r} for {cls.__name__}. Available: {available}"
            logging.error(error_msg)
            raise ValueError(error_msg) from None

    def run(self) -> Results:
        """Resolve the selected components and execute the pipeline.

        Returns:
            Results: Predictions, evaluations, interpretations and the configuration that
            produced them.
        """
        resolved = {stage: self._resolve_stage(stage, component) for stage, component in self.components.items()}
        output = self._execute(resolved)
        return Results(
            config=self.components.copy(),
            task=self.task,
            _submitter=self.submitter,
            **output,
        )

    def _execute(self, resolved: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one run of the resolved components.

        This is the extension point for a benchmark whose protocol differs, such as
        cross-validation or a pretrain/fine-tune schedule: override this and keep the resolution
        and result assembly in :meth:`run`.

        Args:
            resolved (dict): The resolved component of each stage.

        Returns:
            dict: ``predictions``, ``evaluations``, ``interpretations`` and ``artifacts``.
        """
        return run_pipeline(**resolved)
