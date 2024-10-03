# vector-search-playground

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A hands-on vector search sandbox: flat indexes with three distance metrics, k-means clustering, and random-projection dimensionality reduction — the concepts behind every vector database, in ~300 readable lines.

## 🚀 Overview

Before reaching for FAISS or a managed vector DB, it pays to understand what they actually do. `vector-search-playground` implements the fundamentals: an in-memory **FlatIndex** with exact brute-force search over cosine / euclidean / inner-product metrics, **k-means** clustering for exploring embedding geometry, and **random projection** for cheap dimensionality reduction. Everything is dependency-free and every operation has typed, explicit failure modes.

## ✨ Features

- **FlatIndex:** exact top-k search with pluggable metric (`cosine`, `euclidean`, `inner_product`)
- **Filter-aware search:** pass `filter_fn` to restrict candidates by key at query time
- **Live deletion:** remove vectors without rebuilding
- **k-means clustering:** deterministic via seed; empty-cluster pruning; early stop on convergence
- **Random projection:** Johnson–Lindenstrauss-style dimension shrink for fast experiments
- **Usage stats:** track vector count, searches, and current metric
- **Typed errors:** `EmptyIndexError` / `DimensionMismatchError` / bad metric — all under `SearchError`
- **Zero dependencies**

## 🚧 Structure

```
vector-search-playground/
├── src/vector_search/
│   ├── __init__.py
│   └── core.py
├── tests/
│   └── test_core.py
├── README.md
└── pyproject.toml
```

## 📦 Installation

```bash
git clone https://github.com/supremeloki/vector-search-playground.git
cd vector-search-playground
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

## 📋 Requirements

- Python 3.11+
- No runtime dependencies

## 🏃 Quick Start

```python
from vector_search import FlatIndex, kmeans, random_projection_reduce

index = FlatIndex(dimension=3, metric="cosine")
index.add_many({
    "fruit": [1.0, 0.1, 0.0],
    "vehicle": [0.0, 0.2, 1.0],
})

hits = index.search([0.9, 0.0, 0.1], k=1)
print(hits[0])

clusters = kmeans(index_vectors, cluster_count=2)
reduced = random_projection_reduce(all_vectors, target_dimension=8)
```

## 🔧 Error Handling

```text
SearchError
├── EmptyIndexError         # search before any add()
└── DimensionMismatchError  # query/vector dimension mismatch
```

## 🧪 Testing

```bash
pytest tests/ -v
```

## 📝 Code Quality

- Full type hints (`X | None` style), frozen neighbors/clusters
- Zero comments — names carry the meaning
- Seeded randomness → reproducible clustering

## 📄 License

MIT — see [LICENSE](LICENSE).

## 👤 Author

**Kooroush Masoumi**

---

⭐ Star this repo if you find it useful!
