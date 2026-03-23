from abc import ABC, abstractmethod
from typing import (
    ClassVar,
    TypedDict,
    Any,
)


class MetricResult(TypedDict):
    """Typed result for a single example from a single metric."""

    value: float
    """The scalar score for this example."""

    metadata: dict[str, Any]
    """Auxiliary information (e.g., per-sub-metric breakdown)."""


class BaseMetric(ABC):
    """Abstract base for all evaluation metrics.

    Subclasses must customize the logic of metric computation and have a unique name.
    """

    metric_name: ClassVar[str]
    """The name of the metric."""

    @abstractmethod
    def compute(
        self,
        predictions: list[str],
        references: list[list[str]],
        **kwargs: Any,
    ) -> list[MetricResult]:
        """Compute per-example metric scores.

        Args:
            predictions (`list[str]`):
                Model outputs, one per example.
            references (`list[list[str]]`):
                Golden answers for each example. Each element is a list of
                acceptable reference strings (multi-reference support).
            **kwargs (`Any`):
                Additional keyword arguments for metric-specific options.

        Returns:
            `list[MetricResult]`:
                One metric result per example.
        """
        ...
