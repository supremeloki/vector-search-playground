import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from vector_search import (
    DimensionMismatchError,
    EmptyIndexError,
    FlatIndex,
    SearchError,
    kmeans,
    random_projection_reduce,
)


@pytest.fixture
def index():
    flat = FlatIndex(dimension=3, metric="cosine")
    flat.add_many({
        "apple": [1.0, 0.0, 0.0],
        "banana": [0.9, 0.1, 0.0],
        "car": [0.0, 0.0, 1.0],
    })
    return flat


def test_add_and_size(index):
    assert index.size == 3


def test_search_finds_closest_first(index):
    results = index.search([1.0, 0.05, 0.0], k=2)
    assert results[0][0] in {"apple", "banana"}
    assert len(results) == 2
    assert results[0][1] <= results[1][1]


def test_dimension_mismatch_rejected(index):
    with pytest.raises(DimensionMismatchError):
        index.search([1.0, 2.0])


def test_invalid_dimension_rejected():
    with pytest.raises(SearchError):
        FlatIndex(dimension=0)


def test_unknown_metric_rejected():
    with pytest.raises(SearchError):
        FlatIndex(dimension=3, metric="manhattan")


def test_empty_index_search_raises():
    with pytest.raises(EmptyIndexError):
        FlatIndex(dimension=2).search([1.0, 1.0])


def test_filter_restricts_candidates(index):
    results = index.search([1.0, 0.0, 0.0], k=3, filter_fn=lambda key: key != "apple")
    keys = [key for key, _ in results]
    assert "apple" not in keys


def test_delete_removes_entry(index):
    assert index.delete("car") is True
    assert index.delete("car") is False
    assert index.size == 2


def test_euclidean_metric_orders_by_distance():
    flat = FlatIndex(dimension=1, metric="euclidean")
    flat.add_many({"near": [5.0], "far": [50.0]})
    results = flat.search([4.0], k=2)
    assert results[0][0] == "near"


def test_kmeans_separates_two_clusters():
    vectors = {
        "a1": [0.0, 0.0], "a2": [0.1, 0.0],
        "b1": [10.0, 10.0], "b2": [10.1, 10.0],
    }
    clusters = kmeans(vectors, cluster_count=2)
    members = {frozenset(c.members) for c in clusters}
    assert frozenset({"a1", "a2"}) in members
    assert frozenset({"b1", "b2"}) in members


def test_kmeans_needs_enough_vectors():
    with pytest.raises(SearchError):
        kmeans({"only": [1.0]}, cluster_count=3)


def test_projection_shrinks_dimensions():
    vectors = {key: [float(i), float(i * 2), float(i * 3)] for i, key in enumerate(["x", "y", "z"])}
    reduced = random_projection_reduce(vectors, target_dimension=2)
    assert all(len(vec) == 2 for vec in reduced.values())
    assert set(reduced) == {"x", "y", "z"}


def test_projection_rejects_growth():
    with pytest.raises(SearchError):
        random_projection_reduce({"a": [1.0]}, target_dimension=5)


def test_stats_track_usage(index):
    before = index.stats.searches
    index.search([1.0, 0.0, 0.0])
    assert index.stats.searches == before + 1
