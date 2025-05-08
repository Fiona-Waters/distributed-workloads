from abc import ABC, abstractmethod
from pymilvus import Collection
from transformers import RagRetriever
import numpy as np
from feast import FeatureStore
from typing import Optional, Callable, Dict, List


class VectorStore(ABC):
    @abstractmethod
    def query(self, query_vector: np.ndarray, top_k: int):
        pass


class MilvusVectorStore(VectorStore):
    def __init__(self, collection_name: str):
        self.collection = Collection(name=collection_name, using="default")

    def query(self, query_vector: np.ndarray, top_k: int = 5):
        results = self.collection.search(
            data=[query_vector],
            anns_field="vector",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=top_k,
            output_fields=["doc_id"]
        )
        hits = results[0]  
        return [
            {
                "doc_id": hit.entity.get("doc_id"),
                "id": hit.id,
                "score": hit.distance, 
            }
            for hit in hits
        ]

    def text_query(self, query: str, top_k: int = 5):
        raise NotImplementedError("text_query is not implemented for MilvusVectorStore.")


class FeastRAGRetriever(RagRetriever):
    VALID_SEARCH_TYPES = {"text", "vector", "hybrid"}

    def __init__(
        self,
        question_encoder_tokenizer,
        question_encoder,
        generator_tokenizer,
        generator_model,
        feast_repo_path: str,
        vector_store: VectorStore,
        search_type: str,
        config=None,
        format_document: Optional[Callable[[Dict], str]] = None,
        **kwargs
    ):
        if search_type.lower() not in self.VALID_SEARCH_TYPES:
            raise ValueError(
                f"Unsupported search_type {search_type}. Must be one of: {self.VALID_SEARCH_TYPES}"
            )

        super().__init__(
            question_encoder_tokenizer=question_encoder_tokenizer,
            generator_tokenizer=generator_tokenizer, config=config,
            **kwargs
        )
        self.question_encoder = question_encoder
        self.generator_model = generator_model
        self.generator_tokenizer = generator_tokenizer
        self.feast = FeatureStore(repo_path=feast_repo_path)
        self.vector_store = vector_store
        self.search_type = search_type.lower()
        self.format_document = format_document or self._default_format_document

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        if self.search_type == "text":
            return self._text_search(query, top_k)
        elif self.search_type == "vector":
            return self._vector_search(query, top_k)
        elif self.search_type == "hybrid":
            text_results = self._text_search(query, top_k)
            vector_results = self._vector_search(query, top_k)
            return self._merge_results(text_results, vector_results, top_k)
        else:
            raise ValueError(f"Unsupported search type: {self.search_type}")

    def _text_search(self, query: str, top_k: int):
        return self.vector_store.text_query(query, top_k)

    # def _vector_search(self, query: str, top_k: int):
    #     query_vector = self.question_encoder.encode(query)
    #     return self.vector_store.query(query_vector, top_k)
        
    def _vector_search(self, query: str, top_k: int) -> List[Dict]:
        query_vector = self.question_encoder.encode(query)
        vector_store_results = self.vector_store.query(query_vector, top_k)
        doc_ids = [res["doc_id"] for res in vector_store_results if "doc_id" in res and res["doc_id"] is not None]

        if doc_ids:
            feature_refs = [
                f"your_feature_table:{feature_name}"
                for feature_name in ["title", "body"]  # Add your feature names
            ]
            feast_features = self.feast.get_online_features(
                entity_rows=[{"doc_id": doc_id} for doc_id in doc_ids],
                feature_refs=feature_refs,
            ).to_dict()

            retrieved_documents = []
            for i, doc_id in enumerate(doc_ids):
                document = {"doc_id": doc_id, "feast_features": {}}
                for feature_name in ["title", "body"]:
                    document["feast_features"][feature_name] = feast_features[feature_name][i]
                retrieved_documents.append(document)
            return retrieved_documents
        else:
            return []

    def _merge_results(self, text_results, vector_results, top_k):
        # Combine, deduplicate by 'doc_id', and return up to top_k
        seen = set()
        combined = []

        def add_unique(results):
            for r in results:
                doc_id = r.get("doc_id") or r.id  # Adjust based on actual result format
                if doc_id not in seen:
                    combined.append(r)
                    seen.add(doc_id)

        # Deduplicate and merge results from both text and vector queries
        add_unique(text_results)
        add_unique(vector_results)

        return combined[:top_k]

    def generate_answer(self, query: str, top_k: int = 5, max_new_tokens: int = 100) -> str:
        documents = self.retrieve(query, top_k=top_k)
        context = "\n\n".join(documents)
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

        input_ids = self.generator_tokenizer(prompt, return_tensors="pt").input_ids
        output_ids = self.generator_model.generate(input_ids, max_new_tokens=max_new_tokens)

        return self.generator_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def _default_format_document(self, doc: dict) -> str:
        # format the returned document/s into format required for RAG

        features = doc.get("feast_features", {})
        title = features.get("title", "")
        body = features.get("body", "")

        return f"Title: {title}\nBody: {body}"
