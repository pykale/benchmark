"""Split-first pipeline execution with duck-typed components

This module contains orchestration only: it calls the methods that the selected components
already provide. It defines no model, no metric and no training loop. Preprocessing and
embedding are fitted on the training partition only, so a data-dependent step cannot see the
evaluation partition.
"""

import inspect
from typing import Any, Callable, Dict, Sequence, Tuple


def _invoke(function: Callable, *args: Any, **context: Any) -> Any:
    """Call ``function`` with the subset of ``context`` that it declares."""
    signature = inspect.signature(function)
    accepts_all = any(parameter.kind == parameter.VAR_KEYWORD for parameter in signature.parameters.values())
    kwargs = context if accepts_all else {key: value for key, value in context.items() if key in signature.parameters}
    positional = [
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    ][: len(args)]
    return function(*args, **{key: value for key, value in kwargs.items() if key not in positional})


def _method(component: Any, names: Sequence[str], *args: Any, **context: Any) -> Any:
    """Call the first method in ``names`` that ``component`` provides."""
    for name in names:
        if hasattr(component, name):
            return _invoke(getattr(component, name), *args, **context)
    if callable(component):
        return _invoke(component, *args, **context)
    raise TypeError(f"{component!r} must be callable or implement one of {names}")


def _transform_all(component: Any, partitions: Dict[str, Any]) -> Dict[str, Any]:
    """Transform every partition that has data."""
    return {key: _method(component, ("transform",), value) for key, value in partitions.items() if value is not None}


def _fit_transform(component: Any, partitions: Dict[str, Any]) -> Dict[str, Any]:
    """Fit ``component`` on the training partition, then transform them all."""
    if component is None:
        return partitions
    train = partitions["train"]
    if hasattr(component, "fit"):
        _invoke(component.fit, train)
    elif hasattr(component, "fit_transform") and hasattr(component, "transform"):
        rest = {key: value for key, value in partitions.items() if key != "train"}
        return {"train": _invoke(component.fit_transform, train), **_transform_all(component, rest)}
    return _transform_all(component, partitions)


def _xy(value: Any) -> Tuple[Any, Any]:
    """Return ``(features, targets)`` for a prepared partition.

    Accepts the shapes produced by the supported stages: an ``(x, y)`` pair, a
    mapping, an object exposing ``.X``/``.y``, or a dataset of items carrying a
    ``target`` attribute. The last case keeps datasets (rather than feature
    matrices) available to predictors that consume them directly.
    """
    if isinstance(value, dict) and "X" in value and "y" in value:
        return value["X"], value["y"]
    if hasattr(value, "X") and hasattr(value, "y"):
        return value.X, value.y
    # Checked before the pair form: a dataset of two items is not an (x, y) pair.
    if _has_targets(value):
        return value, [_scalar_target(item.target) for item in value]
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return value[0], value[1]
    raise TypeError(
        "Prepared data must expose features and targets as (X, y), {'X', 'y'}, .X/.y, or items with .target"
    )


def _has_targets(value: Any) -> bool:
    """Return whether ``value`` is a dataset whose items carry their own target."""
    try:
        # A data frame indexes columns, not rows, so it is not a dataset of records here.
        return bool(len(value)) and hasattr(value[0], "target")
    except (AttributeError, KeyError, IndexError, TypeError):
        return False


def _scalar_target(target: Any) -> float:
    """Convert a one-value target to a Python float without a numeric-library dependency."""
    if hasattr(target, "item"):
        try:
            return float(target.item())
        except (RuntimeError, ValueError):
            pass
    if isinstance(target, (list, tuple)) and len(target) == 1:
        return float(target[0])
    return float(target)


def _name(component: Any, index: int) -> str:
    """Return a stable result key for a component."""
    return (
        getattr(component, "name", None)
        or getattr(component, "__name__", None)
        or f"{type(component).__name__}_{index}"
    )


def unique_label(label: str, taken: Any) -> str:
    """Return ``label``, suffixed if needed, so that two components never share a result key.

    Args:
        label (str): The label a component asks for.
        taken: The labels already used.

    Returns:
        str: A label not present in ``taken``.
    """
    candidate, suffix = label, 1
    while candidate in taken:
        suffix += 1
        candidate = f"{label}_{suffix}"
    return candidate


def _labelled(value: Any) -> Dict[str, Any]:
    """Return ``{label: component}`` for a single component, list or mapping.

    A mapping is used as given, which is how :class:`~kalebenchmark.benchmark.Benchmark`
    preserves the built-in alias a user asked for (``"mae"`` rather than the
    underlying ``mean_absolute_error``).
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        return {label: component for label, component in value.items() if component is not None}
    components = list(value) if isinstance(value, (list, tuple)) else [value]
    components = [component for component in components if component is not None]
    labelled: Dict[str, Any] = {}
    for index, component in enumerate(components):
        labelled[unique_label(_name(component, index), labelled)] = component
    return labelled


def _prepare(dataset: Any, splitter: Any, prepdata: Any, embed: Any) -> Tuple[Any, Dict[str, Any]]:
    """Load, split, then fit preparation and embedding on the training partition only."""
    data = _method(dataset, ("load", "get_data"))
    partitions = _method(splitter, ("split",), data)
    if not isinstance(partitions, dict) or "train" not in partitions:
        raise TypeError("splitter must return a mapping containing 'train'")
    return data, _fit_transform(embed, _fit_transform(prepdata, partitions))


def _evaluation_key(partitions: Dict[str, Any]) -> str:
    """Return the partition predictions are scored on."""
    for key in ("test", "validation"):
        if key in partitions:
            return key
    raise ValueError("splitter must provide a validation or test partition")


def _collapse(results: Dict[str, Any], single: bool) -> Any:
    """Unwrap the single-predictor case, so one model does not report a nested mapping."""
    return next(iter(results.values())) if single else results


def run(
    *,
    dataset: Any,
    splitter: Any,
    prepdata: Any = None,
    embed: Any = None,
    predict: Any = None,
    evaluate: Any = None,
    interpret: Any = None,
) -> Dict[str, Any]:
    """Execute the benchmark stages in their defined order.

    The sequential path is ``dataset -> splitter -> prepdata -> embed -> predict``. Evaluation and
    interpretation branch from the resulting predictions.

    Args:
        dataset: Component providing ``load()``.
        splitter: Component providing ``split(data)``.
        prepdata: Optional component providing ``fit``/``transform``.
        embed: Optional component providing ``fit``/``transform``.
        predict: Predictor or list of predictors providing ``fit``/``predict``.
        evaluate: Metric or list of metrics called as ``(y_true, y_pred)``.
        interpret: Optional interpreter or list of interpreters.

    Returns:
        dict: ``predictions``, ``evaluations``, ``interpretations`` and ``artifacts``. The first
        three are keyed by predictor name when more than one predictor is given.
    """
    data, partitions = _prepare(dataset, splitter, prepdata, embed)
    x_train, y_train = _xy(partitions["train"])
    x_eval, y_eval = _xy(partitions[_evaluation_key(partitions)])
    predictors, metrics, interpreters = _labelled(predict), _labelled(evaluate), _labelled(interpret)

    predictions, evaluations, interpretations = {}, {}, {}
    for name, predictor in predictors.items():
        _method(predictor, ("fit", "train"), x_train, y_train)
        predictions[name] = _method(predictor, ("predict", "infer"), x_eval)
        context = dict(
            model=predictor,
            pipeline=predictor,
            data=data,
            splits=partitions,
            representations=x_eval,
            predictions=predictions[name],
            targets=y_eval,
        )
        evaluations[name] = {
            label: _method(metric, ("evaluate", "compute"), y_eval, predictions[name], **context)
            for label, metric in metrics.items()
        }
        context["evaluations"] = evaluations[name]
        interpretations[name] = {
            label: _method(interpreter, ("interpret", "explain"), **context)
            for label, interpreter in interpreters.items()
        }

    single = len(predictors) == 1
    return {
        "predictions": _collapse(predictions, single),
        "evaluations": _collapse(evaluations, single),
        "interpretations": _collapse(interpretations, single),
        "artifacts": {"predictors": predictors},
    }
