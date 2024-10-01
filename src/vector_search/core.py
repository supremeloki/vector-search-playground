from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Iterator, Sequence

Vector = tuple[float, ...]


class SearchError(Exception):
    pass


class EmptyIndexError(SearchError):
    pass


class DimensionMismatchError(SearchError):
    def __init__(self, expected: int, actual: int) -> None:
        super().__init__(f"index dimension is {expected}, query has {actual}")


DistanceFn = Callable[[Sequence[float], Sequence[float]], float]


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0
    return 1.0 - (dot / (norm_a * norm_b))


def inner_product_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return -sum(x * y for x, y in zip(a, b))


METRICS: dict[str, DistanceFn] = {
    "euclidean": euclidean,
    "cosine": cosine_distance,
    "inner_product": inner_product_distance,
}


@dataclass(frozen=True)
class Neighbor:
    key: str
    distance: float

    @property
    def rank(self) -> int:
        return self._rank

    def with_rank(self, rank: int) -> "Neighbor":
        object.__setattr__(self, "_rank", rank)
        return self


@dataclass
class IndexStats:
    vectors: int = 0
    dimension: int = 0
    searches: int = 0
    metric: str = "cosine"


class FlatIndex:
    def __init__(self, dimension: int, metric: str = "cosine") -> None:
        if dimension < 1:
            raise SearchError("dimension must be >= 1")
        if metric not in METRICS:
            raise SearchError(f"unknown metric: {metric!r}")
        self._vectors: dict[str, Vector] = {}
        self._dimension = dimension
        self._metric_name = metric
        self.stats = IndexStats(dimension=dimension, metric=metric)

    @property
    def size(self) -> int:
        return len(self._vectors)

    def add(self, key: str, vector: Sequence[float]) -> None:
        if len(vector) != self._dimension:
            raise DimensionMismatchError(self._dimension, len(vector))
        self._vectors[key] = tuple(vector)
        self.stats.vectors = len(self._vectors)

    def add_many(self, pairs: dict[str, Sequence[float]]) -> None:
        for key, vector in pairs.items():
            self.add(key, vector)

    def search(
        self,
        query: Sequence[float],
        k: int = 5,
        filter_fn: Callable[[str], bool] | None = None,
    ) -> list[tuple[str, float]]:
        if not self._vectors:
            raise EmptyIndexError("no vectors indexed")
        if len(query) != self._dimension:
            raise DimensionMismatchError(self._dimension, len(query))
        if k < 1:
            raise SearchError("k must be >= 1")
        distance = METRICS[self._metric_name]
        scored = [
            (key, distance(query, vector))
            for key, vector in self._vectors.items()
            if filter_fn is None or filter_fn(key)
        ]
        scored.sort(key=lambda pair: pair[1])
        self.stats.searches += 1
        return scored[:k]

    def delete(self, key: str) -> bool:
        return self._vectors.pop(key, None) is not None


@dataclass
class Cluster:
    centroid: Vector
    members: list[str] = field(default_factory=list)


def kmeans(
    vectors: dict[str, Sequence[float]],
    cluster_count: int,
    iterations: int = 25,
    seed: int = 7,
) -> list[Cluster]:
    if cluster_count < 1:
        raise SearchError("cluster_count must be >= 1")
    if len(vectors) < cluster_count:
        raise SearchError("not enough vectors for requested clusters")
    keys = sorted(vectors)
    rng = random.Random(seed)
    centroids: list[Vector] = [
        tuple(vectors[key]) for key in rng.sample(keys, cluster_count)
    ]
    assignment: dict[str, int] = {}
    for _ in range(iterations):
        clusters_changed = False
        assignment = {}
        for key in keys:
            distances = [euclidean(vectors[key], c) for c in centroids]
            nearest = distances.index(min(distances))
            if assignment.get(key) != nearest:
                clusters_changed = True
            assignment[key] = nearest
        for index in range(cluster_count):
            member_vectors = [vectors[k] for k in keys if assignment[k] == index]
            if member_vectors:
                dim = len(member_vectors[0])
                centroids[index] = tuple(
                    sum(row[d] for row in member_vectors) / len(member_vectors)
                    for d in range(dim)
                )
        if not clusters_changed:
            break
    result: list[Cluster] = [Cluster(centroid=c) for c in centroids]
    for key in keys:
        result[assignment[key]].members.append(key)
    return [cluster for cluster in result if cluster.members]


def random_projection_reduce(
    vectors: dict[str, Sequence[float]],
    target_dimension: int,
    seed: int = 13,
) -> dict[str, Vector]:
    if target_dimension < 1:
        raise SearchError("target_dimension must be >= 1")
    sample = next(iter(vectors.values()), None)
    if sample is None:
        raise EmptyIndexError("no vectors provided")
    source_dim = len(sample)
    if target_dimension > source_dim:
        raise SearchError("target dimension exceeds source")
    rng = random.Random(seed)
    projection = [
        [rng.gauss(0.0, 1.0 / math.sqrt(target_dimension)) for _ in range(source_dim)]
        for _ in range(target_dimension)
    ]

    def project(vector: Sequence[float]) -> Vector:
        return tuple(
            sum(row[i] * vector[i] for i in range(source_dim))
            for row in projection
        )

    return {key: project(vector) for key, vector in vectors.items()}
