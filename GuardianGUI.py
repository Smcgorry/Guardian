import streamlit as st
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.instructor import InstructorEmbedding
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import ServiceContext, VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

# Choice of LLM via LM Studio Instance
llm = OpenAILike(
    model="mistral-7b-instruct-v0.2.Q6_K.gguf",
    api_key="fake",
    api_base="http://localhost:1234/v1",
    is_chat_model=True
)
# Choice of Embedding Model
embed_model = InstructorEmbedding(model_name="hkunlp/instructor-large", device='cuda')

def process_query(collection_name, question):

    # Load from disk
    db2 = chromadb.PersistentClient(path="/home/headquarters/Documents/Guardian/VectorDatabase/")
    chroma_collection = db2.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
    index = VectorStoreIndex.from_vector_store(vector_store, service_context=service_context)
    # Retrieve Embeddings from Loaded Store, pick size of similarity search('similarity_top_k=')
    retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
    response_synthesizer = get_response_synthesizer(service_context=service_context)
    query_engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=response_synthesizer)

    # Queries Embeddings Retrieved
    response = query_engine.query(question)
    return response

def main():
    # GUI StandUp
    st.title("Guardian")

    collection_name = st.text_input("Enter the name of the collection to pull:")
    question = st.text_area("Write a query for the chatbot:")

    if st.button("Submit"):
        with st.spinner("Processing..."):
            response = process_query(collection_name, question)
            st.success(response)

if __name__ == '__main__':
    main()