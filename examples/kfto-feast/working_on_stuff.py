from transformers.models.retrieval_rag.index_retrieval import Index
from feast import FeatureStore
from pymilvus import Collection
import numpy as np
from typing import Tuple, List

class FeastMilvusIndex(Index):
    def __init__(self, feast_repo_path: str, milvus_collection_name: str):
        self.feast = FeatureStore(repo_path=feast_repo_path)
        self.milvus_collection = Collection(name=milvus_collection_name)
        self._is_initialized = True
    
    def _is_initialized(self):
        return self._is_initialized
    
    def init_index(self):
        self._is_initialized = True

    def get_top_docs(self, question_hidden_states: np.ndarray, n_docs: int) -> Tuple[List[List[str]], List[List[float]]]:
        # Perform search in Milvus        
        search_results = self.milvus_collection.search(
            data=question_hidden_states,
            anns_field="vector",  # update this to name of vector field
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=n_docs,
            output_fields=["doc_id"] # update this depending on stored fields
        )

        # Collect doc_ids in a flat list to batch query Feast
        batched_doc_ids = [
        hit.entity.get("doc_id")
        for hits in search_results
        for hit in hits
    ]

        # Query Feast for document text (or other metadata)
        feast_results = self.feast.get_online_features(
            features=[
                # what are our feature definitions?
            ],
            entity_rows=[{"doc_id": doc_id} for doc_id in batched_doc_ids]
        ).to_dict()

        # Map doc_id to text
        doc_id_to_text = {
            batched_doc_ids[i]: feast_results["docs:text"][i]
            for i in range(len(batched_doc_ids))
        }

        # Rebuild the batched result format
        docs_texts = []
        docs_scores = []

        for hits in search_results:
            texts = []
            scores = []

            for hit in hits:
                doc_id = hit.entity.get("doc-id")
                text = doc_id_to_text.get(doc_id, "")
                score = hit.score
        
                texts.append(text)
                scores.append(score)
            
            docs_texts.append(texts)
            docs_scores.append(scores)

        return docs_texts, docs_scores
    
    def get_doc_dicts(self, doc_ids: np.ndarray) -> List[dict]:
        # Implemetation of this function is optional, can be used for debugging.

        # Flatten the input doc_ids array
        flat_doc_ids = doc_ids.flatten().tolist()

        # Query Feast for document metadata
        feast_response = self.feast.get_online_features(
            features=["docs:title", "docs:text"],
            entity_rows=[{"doc_id": doc_id} for doc_id in flat_doc_ids]
        ).to_dict()

        # Assemble dicts (1 per doc)
        doc_dicts = []
        for i in range(len(flat_doc_ids)):
            doc_dicts.append({
                "title": feast_response["docs:title"][i],
                "text": feast_response["docs:text"][i]
            })

        return doc_dicts