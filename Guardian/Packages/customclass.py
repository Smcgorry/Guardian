from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import TextNode
from chromadb import EmbeddingFunction, Documents, Embeddings

class LlamaIndexEmbeddingAdapter(EmbeddingFunction):
    def __init__(self, ef:BaseEmbedding):
        self.ef = ef

    def __call__(self, input: Documents) -> Embeddings:
        return [node.embedding for node in self.ef([TextNode(text=doc) for doc in input])]