import os
import pickle
import torch
import faiss
import gc
import logging
from typing import List, Union, Optional, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)

class TopkRetriever:
    def __init__(self, 
                 database: List[Tuple[str, str, str]],
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 top_k: int = 5,
                 threshold: float = 0.6,
                 batch_size: int = 2048,
                 cache_dir: Optional[str] = "./database_cache",
                 database_name: str = "default_db",
                 verbose: bool = True):
        """
        Args:
            database: List of (entity, relation, value) triplets
            model_name: SentenceTransformer model path or name
            top_k: Number of nearest neighbors to retrieve
            threshold: Similarity threshold
            batch_size: Batch size for encoding
            cache_dir: Directory to cache FAISS index and mappings
            database_name: Name used for cache files
            verbose: Whether to log info
        """
        self.database = database
        self.top_k = top_k if top_k else 5
        self.default_threshold = threshold
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.database_name = database_name
        self.verbose = verbose

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model = self.model.half().eval()

        self.index = None
        self.id_to_triplet = {}

        if self.cache_dir and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

        cache_path = self._get_cache_path()

        if cache_path and os.path.exists(f"{cache_path}.index") and os.path.exists(f"{cache_path}.mapping"):
            self._load_from_cache(cache_path)
            if self.verbose:
                logger.info(f"Loaded FAISS index of {len(self.id_to_triplet)} triplets from {cache_path}")
        else:
            self._build_index()
            if cache_path:
                self._save_to_cache(cache_path)
                if self.verbose:
                    logger.info(f"Saved FAISS index of {len(self.id_to_triplet)} triplets to {cache_path}")

        gc.collect()
        torch.cuda.empty_cache()

    def _get_cache_path(self):
        if not self.cache_dir:
            return None
        return os.path.join(self.cache_dir, f"{self.database_name}_{len(self.database)}")

    def _save_to_cache(self, cache_path):
        faiss.write_index(self.index, f"{cache_path}.index")
        with open(f"{cache_path}.mapping", 'wb') as f:
            pickle.dump(self.id_to_triplet, f)

    def _load_from_cache(self, cache_path):
        self.index = faiss.read_index(f"{cache_path}.index")
        with open(f"{cache_path}.mapping", 'rb') as f:
            self.id_to_triplet = pickle.load(f)

    def _build_index(self):
        embedding_dim = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(embedding_dim))

        texts = [f"{self._normalize_text(ent)} {self._normalize_text(rel)}" for ent, rel, _ in self.database]
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=self.verbose
        )

        ids = np.arange(len(self.database))
        self.index.add_with_ids(embeddings, ids)
        self.id_to_triplet = {i: triplet for i, triplet in enumerate(self.database)}

    def retrieve_top_k(self, entity: str, relation: str, threshold: Optional[float] = None) -> List[str]:
        query_text = f"{self._normalize_text(entity)} {self._normalize_text(relation)}"
        query_embedding = self.model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)

        distances, indices = self.index.search(query_embedding, self.top_k)

        # Use per-query threshold if passed, otherwise fallback to default
        th = threshold if threshold is not None else self.default_threshold

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx in self.id_to_triplet and dist >= th:
                assert 1.01 > dist >= -1.01, f"Distance {dist} is out of bounds"

                triplet = self.id_to_triplet[idx]
                results.append((triplet[0], triplet[1], triplet[2], float(dist)))

        results.sort(key=lambda x: x[-1], reverse=True)
        # logger.info(f"retrieve_top_k Threshold: {th}, Results: {results}")
        return [r[2] for r in results]

    @staticmethod
    def _normalize_text(text: str) -> str:
        return text.lower().replace("_", " ").strip()


if __name__ == "__main__":
    import json

    # Load your database
    database_path = "./data/database/dwiki-eval100-annotator_database.json"

    with open(database_path, 'r') as f:
        data = json.load(f)
    triplets = data["triplets"]

    retriever = TopkRetriever(
        database=triplets,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_dir="./database_cache",
        database_name="dwiki_bio17k-annotator",
        top_k=5,
        verbose=True
    )

    queries = [
        ("earth", "diameter"),
        ("earth", "number of moons"),
        ("Walter Luis Corbo Burmia", "occupation"),
    ]

    # test threshold
    th_lst = [None, 0.6, 0.8]
    for th in th_lst:
        print(f"Threshold: {th}")
        for entity, relation in queries:
            results = retriever.retrieve_top_k(entity, relation, threshold=th)
            print(f"Query: ({entity}, {relation}) -> {results}")
