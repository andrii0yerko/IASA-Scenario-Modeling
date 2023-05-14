import logging
from recordtype import recordtype
from pathlib import Path

import numpy as np
import pandas as pd
import pymongo.cursor
import yaml
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier

from database import FilmsDB

SearchResult = recordtype("SearchResult", "result,embedding")
KNN_Data = recordtype("KNN_Data", "X,y,ids")

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s", force=True)


class EmbeddingSearch:
    def __init__(self, vectors: np.ndarray, embedder: callable, indexes: np.ndarray = None):
        self._vectors = vectors
        self.embedder = embedder
        self.indexes = indexes

    @classmethod
    def from_texts(cls, inputs: list[str], embedder: callable):
        _vectors = cls._create_db(inputs, embedder)
        return cls(np.array(_vectors), embedder)

    @classmethod
    def from_database(cls, database: FilmsDB, embedder):
        _indexes, _vectors = cls._create_embeddings(database.get_all_keywords(), embedder, database.count_all())
        return cls(np.array(_vectors), embedder, np.array(_indexes))

    @staticmethod
    def _create_embeddings(inputs: pymongo.cursor.Cursor, embedder, total):
        logging.debug("creating keyword embeddings")
        result = []
        indexes = []
        step = total // 100
        for i, doc in enumerate(inputs):
            index = str(doc['_id'])
            text = doc['keywords']
            vec = embedder(text)
            result.append(vec)
            indexes.append(index)
            if i % step == 0:
                logging.debug("%s/%s", i, total)
        return indexes, result

    @staticmethod
    def _create_db(inputs: list[str], embedder):
        logging.debug("creating db")
        result = []
        total = len(inputs)
        step = total // 100
        for i, text in enumerate(inputs):
            vec = embedder(text)
            result.append(vec)
            if i % step == 0:
                logging.debug("%s/%s", i, total)
        return result

    def from_pickle(self, path):
        pass

    def get_closest(self, query: str, n: int = 1000) -> list[dict]:
        query_vec = self.embedder(query)
        dist = pairwise_distances(query_vec[None, ...], self._vectors, "cosine")
        dist = dist.ravel()
        idx = np.argsort(dist)[:n]

        if self.indexes is not None:
            result = [{"id": self.indexes[_id],
                       "distance": dist,
                       "relevance": None}
                      for _id, dist in zip(idx, dist[idx])]
        else:
            result = [{"id": _id,
                       "distance": dist,
                       "relevance": None}
                      for _id, dist in zip(idx, dist[idx])]

        return SearchResult(result, query_vec)

    def get_rerank(self, labeling: list[dict]):
        pass

    @property
    def vectors(self):
        return self._vectors


class KNN_Marker:
    def __init__(self, model_params: dict = None):
        if model_params is None:
            model_params = {
                'weights': 'distance',
                'algorithm': 'auto',
                'metric': 'cosine'
            }
        self.model = KNeighborsClassifier(**model_params)

    @staticmethod
    def _data_extraction(results: SearchResult, embeddings: EmbeddingSearch) -> dict:
        keywords_embeddings_unmarked = []
        keywords_embeddings_marked = [results.embedding]

        marked_values = [1]

        marked_ids = [-1]
        unmarked_ids = []

        for result in results.result:
            if embeddings.indexes is not None:
                _id = embeddings.indexes.index(result["id"])
            else:
                _id = result["id"]
            embedding = embeddings.vectors[_id]
            if result["relevance"] is None:
                keywords_embeddings_unmarked.append(embedding)
                unmarked_ids.append(result["id"])
            elif result["relevance"]:
                keywords_embeddings_marked.append(embedding)
                marked_values.append(1)
                marked_ids.append(result["id"])
            elif not result["relevance"]:
                keywords_embeddings_marked.append(embedding)
                marked_values.append(0)
                marked_ids.append(result["id"])

        data = {
            "train": KNN_Data(np.array(keywords_embeddings_marked), np.array(marked_values), marked_ids),
            "test": KNN_Data(np.array(keywords_embeddings_unmarked), (), unmarked_ids)
        }

        return data

    def _knn_marking(self, data: dict) -> dict:
        self.model.fit(data["train"].X, data["train"].y)
        y_test = self.model.predict(data["test"].X)
        data["test"].y = y_test
        return data

    @staticmethod
    def _results_update(data: dict, results: SearchResult, embeddings: EmbeddingSearch) -> SearchResult:
        for result in results.result:
            if embeddings.indexes is not None:
                _id = embeddings.indexes.index(result["id"])
            else:
                _id = result["id"]

            if result["relevance"] is not None:
                continue

            index = data["test"].ids.index(result["id"])
            relevance = data["test"].y[index]
            result["relevance"] = bool(relevance)
        return results

    def knn_marker(self, results: SearchResult, embeddings: EmbeddingSearch) -> SearchResult:
        data = self._data_extraction(results, embeddings)
        data = self._knn_marking(data)
        results = self._results_update(data, results, embeddings)
        return results

    def model_performance(self):
        pass
