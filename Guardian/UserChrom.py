import os
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, get_response_synthesizer, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
#from llama_index.core.postprocessor import SimilarityPostprocessor
#from llama_index.embeddings.instructor import InstructorEmbedding
from llama_index.llms.openai_like import OpenAILike
import chromadb
from langchain_community.embeddings import GPT4AllEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.packs.corrective_rag import CorrectiveRAGPack

#Load LLM
Settings.llm = OpenAILike(
    model="phi-2-dpo.Q8_0.gguf",
    api_key="fake",
    api_base="http://localhost:1234/v1",
    is_chat_model=True
)
Settings.embed_model = LangchainEmbedding(GPT4AllEmbeddings(model='http://localhost:1234/v1/embeddings'))
whoosh = "/home/headquarters/Documents/Guardian/WebIndex/"

collection_name = input("Enter the name of the collection to pull: ")
question = input("Write a query for the chatbot: ")

# load from disk
db2 = chromadb.PersistentClient(path="/home/headquarters/Documents/Guardian/VectorDatabase/")

chroma_collection = db2.get_or_create_collection(collection_name)

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

index = VectorStoreIndex.from_vector_store(
    vector_store,
    embed_model=Settings.embed_model
)

# Retrieves Documents
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=3,
)

response_synthesizer = get_response_synthesizer(llm=Settings.llm)

#Query code configuration
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer
)

#Regular Chat Output:
corrective_rag= CorrectiveRAGPack(vector_store,Settings.llm,whoosh)
print('-' * 100)
print("The response of the query " + question + " is:")
response= corrective_rag.run(question, similarity_top_k=2)
print('-' * 100)
print(response)