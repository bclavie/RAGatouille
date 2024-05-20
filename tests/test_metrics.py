from ragatouille.evaluation.metrics import *


def test_hit_rate():
    hit_rate_metric = HitRate()

    # Typical case: Both expected and retrieved IDs are provided, and at least one ID matches
    expected_ids = ["1", "2", "3", "4"]
    retrieved_ids = ["2", "5", "6"]
    assert hit_rate_metric.compute(expected_ids, retrieved_ids) == 1.0

    # Typical case: Both expected and retrieved IDs are provided, but no match
    expected_ids = ["1", "2", "3", "4"]
    retrieved_ids = ["5", "6"]
    assert hit_rate_metric.compute(expected_ids, retrieved_ids) == 0.0

    # Edge case: Both lists are empty
    expected_ids = []
    retrieved_ids = []
    assert hit_rate_metric.compute(expected_ids, retrieved_ids) == 0.0

    # Edge case: No expected IDs provided
    expected_ids = None
    retrieved_ids = ["5", "6"]
    try:
        hit_rate_metric.compute(expected_ids, retrieved_ids)
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Edge case: No retrieved IDs provided
    expected_ids = ["1", "2", "3", "4"]
    retrieved_ids = None
    try:
        hit_rate_metric.compute(expected_ids, retrieved_ids)
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Edge case: Both lists are None
    expected_ids = None
    retrieved_ids = None
    try:
        hit_rate_metric.compute(expected_ids, retrieved_ids)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_recall():
    recall_metric = Recall()

    # Typical case: Both expected and retrieved IDs are provided
    expected_ids = ["1", "2", "3", "4"]
    retrieved_ids = ["2", "3", "5", "6"]
    assert recall_metric.compute(expected_ids, retrieved_ids) == 2 / 4

    # Edge case: Both lists are empty
    expected_ids = []
    retrieved_ids = []
    assert recall_metric.compute(expected_ids, retrieved_ids) == 0.0

    # Edge case: No relevant IDs retrieved
    expected_ids = ["1", "2", "3", "4"]
    retrieved_ids = ["5", "6", "7", "8"]
    assert recall_metric.compute(expected_ids, retrieved_ids) == 0.0

    # Edge case: No expected IDs provided
    expected_ids = None
    retrieved_ids = ["5", "6", "7", "8"]
    try:
        recall_metric.compute(expected_ids, retrieved_ids)
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Edge case: No retrieved IDs provided
    expected_ids = ["1", "2", "3", "4"]
    retrieved_ids = None
    try:
        recall_metric.compute(expected_ids, retrieved_ids)
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Edge case: Both lists are None
    expected_ids = None
    retrieved_ids = None
    try:
        recall_metric.compute(expected_ids, retrieved_ids)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_mrr():
    mrr_metric = MRR()

    # Typical case: Both expected and retrieved IDs are provided, and the first ID matches
    expected_ids = ["1", "2", "3", "4"]
    retrieved_ids = ["2", "5", "6"]
    assert mrr_metric.compute(expected_ids, retrieved_ids) == 1.0

    # Typical case: Both expected and retrieved IDs are provided, and the second ID matches
    expected_ids = ["1", "2", "3", "4"]
    retrieved_ids = ["5", "2", "6"]
    assert mrr_metric.compute(expected_ids, retrieved_ids) == 0.5

    # Typical case: Both expected and retrieved IDs are provided, but no match
    expected_ids = ["1", "2", "3", "4"]
    retrieved_ids = ["5", "6"]
    assert mrr_metric.compute(expected_ids, retrieved_ids) == 0.0

    # Edge case: Both lists are empty
    expected_ids = []
    retrieved_ids = []
    assert mrr_metric.compute(expected_ids, retrieved_ids) == 0.0

    # Edge case: No expected IDs provided
    expected_ids = None
    retrieved_ids = ["5", "6"]
    try:
        mrr_metric.compute(expected_ids, retrieved_ids)
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Edge case: No retrieved IDs provided
    expected_ids = ["1", "2", "3", "4"]
    retrieved_ids = None
    try:
        mrr_metric.compute(expected_ids, retrieved_ids)
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Edge case: Both lists are None
    expected_ids = None
    retrieved_ids = None
    try:
        mrr_metric.compute(expected_ids, retrieved_ids)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_resolve_metrics():
    # Test with valid metric names
    valid_metrics = ["recall", "hit_rate", "mrr"]
    resolved_metrics = resolve_metrics(valid_metrics)
    assert len(resolved_metrics) == len(valid_metrics)
    for metric, resolved_metric_class in zip(valid_metrics, resolved_metrics):
        assert resolved_metric_class == METRIC_REGISTRY[metric]

    # Test with invalid metric name
    invalid_metric = ["invalid_metric"]
    try:
        resolve_metrics(invalid_metric)
        assert False, "Expected ValueError for invalid metric name"
    except ValueError:
        pass

    # Test with empty list of metrics
    empty_metrics = []
    resolved_empty_metrics = resolve_metrics(empty_metrics)
    assert len(resolved_empty_metrics) == 0
