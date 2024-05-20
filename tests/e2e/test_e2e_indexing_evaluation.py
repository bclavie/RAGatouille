import srsly

from ragatouille import RAGPretrainedModel


def test_indexing():
    RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
    with open("tests/data/miyazaki_wikipedia.txt", "r") as f:
        full_document = f.read()
    RAG.index(
        collection=[full_document],
        index_name="Miyazaki",
        max_document_length=180,
        split_documents=True,
    )
    # ensure collection is stored to disk
    collection = srsly.read_json(
        ".ragatouille/colbert/indexes/Miyazaki/collection.json"
    )
    assert len(collection) > 1


def test_evaluate():
    RAG = RAGPretrainedModel.from_index(".ragatouille/colbert/indexes/Miyazaki/")
    k = 3

    results = RAG.search(query="What animation studio did Miyazaki found?", k=k)
    assert len(results) == k

    metrics = ["hit_rate", "recall", "mrr"]
    metric_dict = RAG.evaluate(
        ["What animation studio did Miyazaki found?"],
        [[result["document_id"] for result in results[:k]]],
        [[result["passage_id"] for result in results[:k]]],
        metrics=metrics,
        k=[k],
    )

    for metric in metrics:
        assert metric_dict[metric][k] == 1.0
