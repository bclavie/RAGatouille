import numpy as np

from ragatouille.RAGPretrainedModel import RAGPretrainedModel


def mock_search_results(
    queries, search_results, model, expected_ids, metrics, k_values
):
    model.model.search = lambda queries, k: search_results
    return model.evaluate(queries, expected_ids, metrics=metrics, k=k_values)


def test_evaluate():
    model = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

    queries = ["query1", "query2"]
    expected_ids = [["doc1", "doc2", "doc3"], ["doc2", "doc3", "doc4"]]

    metrics = ["recall", "hit_rate", "mrr"]
    k_values = [1, 2, 3]

    search_results1 = [
        [
            {"document_id": "doc1", "rank": 1},
            {"document_id": "doc2", "rank": 2},
            {"document_id": "doc3", "rank": 3},
        ],
        [
            {"document_id": "doc2", "rank": 1},
            {"document_id": "doc3", "rank": 2},
            {"document_id": "doc4", "rank": 3},
        ],
    ]

    result1 = mock_search_results(
        queries, search_results1, model, expected_ids, metrics, k_values
    )

    expected_results1 = {
        "recall": {1: 0.3333333333333333, 2: 0.6666666666666666, 3: 1.0},
        "hit_rate": {
            1: 1.0,
            2: 1.0,
            3: 1.0,
        },
        "mrr": {1: 1.0, 2: 1.0, 3: 1.0},
    }

    for metric_name, metric_data in result1.items():
        for k, value in metric_data.items():
            assert np.isclose(value, expected_results1[metric_name][k])

    search_results2 = [
        [
            {"document_id": "doc1", "rank": 1},
            {"document_id": "doc2", "rank": 2},
            {"document_id": "doc3", "rank": 3},
        ],
        [
            {"document_id": "doc5", "rank": 1},
            {"document_id": "doc6", "rank": 2},
            {"document_id": "doc7", "rank": 3},
        ],
    ]

    result2 = mock_search_results(
        queries, search_results2, model, expected_ids, metrics, k_values
    )

    expected_results2 = {
        "recall": {1: 0.1666666666666666, 2: 0.3333333333333333, 3: 0.5},
        "hit_rate": {
            1: 0.5,
            2: 0.5,
            3: 0.5,
        },
        "mrr": {
            1: 0.5,
            2: 0.5,
            3: 0.5,
        },
    }

    for metric_name, metric_data in result2.items():
        for k, value in metric_data.items():
            assert np.isclose(value, expected_results2[metric_name][k])

    search_results3 = [
        [
            {"document_id": "doc1", "rank": 1},
            {"document_id": "doc2", "rank": 2},
            {"document_id": "doc3", "rank": 3},
        ],
        [
            {"document_id": "doc6", "rank": 1},
            {"document_id": "doc5", "rank": 2},
            {"document_id": "doc4", "rank": 3},
        ],
    ]

    result3 = mock_search_results(
        queries, search_results3, model, expected_ids, metrics, k_values
    )

    expected_results3 = {
        "recall": {1: 0.1666666666666666, 2: 0.3333333333333333, 3: 0.6666666666666666},
        "hit_rate": {
            1: 0.5,
            2: 0.5,
            3: 1.0,
        },
        "mrr": {
            1: 0.5,
            2: 0.5,
            3: 0.6666666666666666,
        },
    }

    for metric_name, metric_data in result3.items():
        for k, value in metric_data.items():
            assert np.isclose(value, expected_results3[metric_name][k])
