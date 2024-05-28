from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.retrievers import VectorIndexRetriever
import re


class QuestionEvaluator:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def get_metadata_ids(self, question):
        # Create the vector index from the vector store
        index = VectorStoreIndex.from_vector_store(
            self.vector_store,
            embed_model=Settings.embed_model
        )

        # Agent Node Processor
        retriever = VectorIndexRetriever(
            index=index
        )

        # Retrieve the nodes
        nodes = retriever.retrieve(question)

        # Extract metadata from nodes
        metadata_ids = [node.metadata for node in nodes]

        return metadata_ids

    def evaluate_response(self, question):
        metadata_ids = self.get_metadata_ids(question)
        question_words = set(re.sub(r'[^\w\s]', '', question.lower()).split())

        # Extract values from metadata dictionaries for comparison
        combined_match_words = set()
        for metadata in metadata_ids:
            combined_match_words.update(metadata.values())

        matches_found = combined_match_words.intersection(question_words)

        if matches_found:
            return True
        else:
            return {
                "match": False,
                "metadata_ids": metadata_ids,
                "response": metadata_ids
            }