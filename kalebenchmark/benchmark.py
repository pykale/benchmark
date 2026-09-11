"""The user-facing benchmark composition API."""

import inspect

from .pipeline import _name
from .pipeline import run as run_pipeline
from .pipeline import unique_label
from .results import Results
from .splitdata.dataset_split import PredefinedSplit, RandomSplit
from .task import TaskCard

STAGES = ("dataset", "splitter", "prepdata", "embed", "predict", "evaluate", "interpret")
MULTI_STAGES = ("predict", "evaluate", "interpret")

GENERIC_BUILTINS = {
    "dataset": {},
    "splitter": {
        "random_split": RandomSplit,
        "predefined_split": PredefinedSplit,
    },
    "prepdata": {},
    "embed": {},
    "predict": {},
    "evaluate": {},
    "interpret": {},
}


def merge_builtins(*registries):
    """Combine shortcut dictionaries stage by stage."""
    merged = {stage: {} for stage in STAGES}
    for registry in registries:
        for stage, components in registry.items():
            merged[stage].update(components)
    return merged


class Benchmark:
    """Run replaceable components in benchmark pipeline order.

    A component can be an object or a name from ``BUILTINS``. Concrete benchmarks only need to
    subclass this class and extend ``BUILTINS``; see ``examples.realmat_bag.RealMatBaG``.
    """

    BUILTINS = GENERIC_BUILTINS
    TASKS: dict[str, TaskCard] = {}

    def __init__(
        self,
        dataset,
        splitter,
        prepdata=None,
        embed=None,
        predict=None,
        evaluate=None,
        interpret=None,
    ):
        self.components = {
            "dataset": dataset,
            "splitter": splitter,
            "prepdata": prepdata,
            "embed": embed,
            "predict": predict,
            "evaluate": evaluate,
            "interpret": interpret,
        }
        self.task = None
        self.submitter = None

    @classmethod
    def resolve(cls, stage, component):
        """Turn a registered name into a component; leave custom objects unchanged."""
        if component is None or not isinstance(component, str):
            return component
        try:
            component = cls.BUILTINS[stage][component]
        except KeyError:
            choices = ", ".join(sorted(cls.BUILTINS.get(stage, {}))) or "none"
            raise ValueError(
                f"Unknown {stage} {component!r} for {cls.__name__}. Available: {choices}. "
                "Discipline-specific shortcuts live on the matching Benchmark subclass."
            ) from None
        return component() if inspect.isclass(component) else component

    @classmethod
    def _resolve_stage(cls, stage, value):
        """Resolve one stage and label stages that accept multiple components."""
        if stage not in MULTI_STAGES:
            return cls.resolve(stage, value)
        if isinstance(value, dict):
            return {name: cls.resolve(stage, component) for name, component in value.items()}

        values = value if isinstance(value, (list, tuple)) else [value]
        resolved = {}
        for index, component in enumerate(values):
            if component is None:
                continue
            item = cls.resolve(stage, component)
            label = component if isinstance(component, str) else _name(item, index)
            resolved[unique_label(label, resolved)] = item
        return resolved

    @classmethod
    def from_task(cls, task, submitter=None, **components):
        """Build a benchmark while protecting the components fixed by a task card."""
        if isinstance(task, str):
            try:
                task = cls.TASKS[task]
            except KeyError:
                choices = ", ".join(sorted(cls.TASKS)) or "none"
                raise ValueError(f"Unknown task {task!r} for {cls.__name__}. Available: {choices}") from None
        if not isinstance(task, TaskCard):
            raise TypeError("task must be a task name or TaskCard")

        fixed = {stage: value for stage, value in task.components().items() if value is not None}
        overridden = sorted(fixed.keys() & components.keys())
        if overridden:
            raise ValueError(f"'{overridden[0]}' is fixed by TaskCard '{task.name}' and cannot be overridden.")

        benchmark = cls(**{**components, **fixed})
        benchmark.task = task
        benchmark.submitter = submitter
        return benchmark

    def run(self):
        """Resolve the selected components and run the pipeline."""
        resolved = {stage: self._resolve_stage(stage, component) for stage, component in self.components.items()}
        output = self._execute(resolved)
        return Results(
            config=self.components.copy(),
            task=self.task,
            _submitter=self.submitter,
            **output,
        )

    def _execute(self, components):
        """Override this method only when a benchmark needs a different execution protocol."""
        return run_pipeline(**components)
