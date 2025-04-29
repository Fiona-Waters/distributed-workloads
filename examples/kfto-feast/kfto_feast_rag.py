from transformers import RagRetriever
import torch
from pymilvus import Collection, connections
from feast import FeatureStore
import numpy as np
 
class FeastRAGRetriever(RagRetriever):
    def __init__(
            self, 
            question_encoder_tokenizer, 
            generator_tokenizer, 
            feast_repo_path: str, 
            milvus_collection_name: str, 
            **kwargs
    ):
        super().__init__(question_encoder_tokenizer=question_encoder_tokenizer,
                          generator_tokenizer=generator_tokenizer, 
                          **kwargs)
        self.feast = FeatureStore(repo_path=feast_repo_path)
        self.milvus_collection = Collection(name=milvus_collection_name)

    def retrieve(self, question_hidden_states: np.ndarray, n_docs: int):
        # Retrieve top-k documents using Milvus and Feast, returning in HF compatible RAG format.
        
        # Convert to numpy 
        # query_embeddings = question_hidden_states.detach().cpu().numpy()
        
        # Perform search in Milvus
        search_results = self.milvus_collection.search(
            data=question_hidden_states,
            anns_field="vector", # update this to name of vector field
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=n_docs,
            output_fields=["doc_id"] # update this depending on stored fields
        )

        # Extract top-k doc_ids for each query
        top_doc_ids = [[hit.entity.get("doc_id") for hit in result] for result in search_results]
        flat_doc_ids = [doc_id for sublist in top_doc_ids for doc_id in sublist]
        
        # Use doc_ids to get metadata from Feast
        feast_results = self.feast.get_online_features(
            features=[
                # what are our feature definitions?
            ],
            entity_rows=[{"doc_id": doc_id } for doc_id in flat_doc_ids]
        ).to_df()

        # Tokenize results returned from Feast as tensors
        context_input_ids = []
        context_attention_mask = []
        doc_scores = []

        for i, result in enumerate(search_results):
            input_ids = []
            attention_masks = []
            scores = []

            for hit in result:
                doc_id = hit.entity.get("doc_id")
                doc_text = feast_results[feast_results["doc_id"] == doc_id]["docs:text"].values[0]
                encoding = self.generator_tokenizer(
                    doc_text,
                    truncation=True,
                    padding="max_length",
                    max_length=self.generator_tokenizer.model_max_length,
                    return_tensors="pt"
                )
                input_ids.append(encoding["input_ids"])
                attention_masks.append(encoding["attention_mask"])
                scores.append(hit.distance)
            
            context_input_ids.append(torch.cat(input_ids))
            context_attention_mask.append(torch.cat(attention_masks))
            doc_scores.append(torch.tensor(scores))
        
        return {
            "context_input_ids": torch.stack(context_input_ids),
            "context_attention_mask": torch.stack(context_attention_mask),
            "doc_scores": torch.stack(doc_scores)
        }

    def retrieve_simple(self, question_hidden_states: np.ndarray, n_docs: int):
        # Retrieve top-k documents using Milvus and Feast, returning a simple dictionary.
        # Convert to numpy 
        # query_embeddings = question_hidden_states.detach().cpu().numpy()

        # Perform search in Milvus
        search_results = self.milvus_collection.search(
            data=question_hidden_states,
            anns_field="vector", # update this to name of vector field
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=n_docs,
            output_fields=["doc_id"] # update this depending on stored fields
        )

        # Collect and format results
        results_dict = {}
        for i, hits in enumerate(search_results):
            doc_ids = [hit.id for hit in hits]
            scores = [hit.score for hit in hits]

            # Use doc_ids to get metadata from Feast
            feast_results = self.feast.get_online_features(
                features=["docs:title", "docs:text"], # update this based on our feature definitions
                entity_rows=[{"doc_id": doc_id} for doc_id in doc_ids]
            ).to_dict()

            docs = [
                {
                    "title": feast_results["docs:title"][j],
                    "text": feast_results["docs:text"][j],
                    "score": scores[j]
                }
                for j in range(len(doc_ids))
            ]

            results_dict[i] = docs

        return results_dict
