from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Tuple, Union


class BaseMetric(ABC):
    """Base class for metrics."""

    metric_name: str

    @abstractmethod
    def compute(
        self,
        expected_ids: Optional[List[Union[str, Tuple[str, int]]]] = None,
        retrieved_ids: Optional[List[Union[str, Tuple[str, int]]]] = None,
        **kwargs: Any,
    ) -> float:
        """Compute metric.

        Args:
            expected_ids (Optional[List[Union[str, Tuple[str, int]]]]): Expected ids
            retrieved_ids (Optional[List[Union[str, Tuple[str, int]]]]): Retrieved ids
            **kwargs: Additional keyword arguments
        """


class HitRate(BaseMetric):
    """Hit rate metric."""

    metric_name: str = "hit_rate"

    def compute(
        self,
        expected_ids: Optional[List[Union[str, Tuple[str, int]]]] = None,
        retrieved_ids: Optional[List[Union[str, Tuple[str, int]]]] = None,
        **kwargs: Any,
    ) -> float:
        """Compute metric."""
        if retrieved_ids is None or expected_ids is None:
            raise ValueError("Retrieved ids and expected ids must be provided")
        is_hit = any(id in expected_ids for id in retrieved_ids)
        return 1.0 if is_hit else 0.0


class Recall(BaseMetric):
    """Recall metric."""

    metric_name: str = "recall"

    def compute(
        self,
        expected_ids: Optional[List[Union[str, Tuple[str, int]]]] = None,
        retrieved_ids: Optional[List[Union[str, Tuple[str, int]]]] = None,
        **kwargs: Any,
    ) -> float:
        """Compute metric."""
        if retrieved_ids is None or expected_ids is None:
            raise ValueError("Retrieved ids and expected ids must be provided")
        num_expected = len(expected_ids)
        num_relevant = len(set(expected_ids) & set(retrieved_ids))
        return num_relevant / num_expected if num_expected > 0 else 0.0


class MRR(BaseMetric):
    """MRR metric."""

    metric_name: str = "mrr"

    def compute(
        self,
        expected_ids: Optional[List[Union[str, Tuple[str, int]]]] = None,
        retrieved_ids: Optional[List[Union[str, Tuple[str, int]]]] = None,
        **kwargs: Any,
    ) -> float:
        """Compute metric."""
        if retrieved_ids is None or expected_ids is None:
            raise ValueError("Retrieved ids and expected ids must be provided")
        for i, id in enumerate(retrieved_ids):
            if id in expected_ids:
                return 1.0 / (i + 1)
        return 0.0


METRIC_REGISTRY: Dict[str, Type[BaseMetric]] = {
    "hit_rate": HitRate,
    "recall": Recall,
    "mrr": MRR,
}


def resolve_metrics(metrics: List[str]) -> List[Type[BaseMetric]]:
    """Resolve metrics from list of metric names."""
    for metric in metrics:
        if metric not in METRIC_REGISTRY:
            raise ValueError(f"Invalid metric name: {metric}")

    return [METRIC_REGISTRY[metric] for metric in metrics]
