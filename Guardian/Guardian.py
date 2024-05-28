import os
import streamlit as st
from llama_index.llms.openai_like import OpenAILike
#from llama_index.embeddings.instructor import InstructorEmbedding
from langchain_community.embeddings import GPT4AllEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, get_response_synthesizer, Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.packs.corrective_rag import CorrectiveRAGPack
from Packages.QuestionEval import QuestionEvaluator
import re
Settings.llm = OpenAILike(
    model="Meta-Llama-3-8B-Instruct.Q8_0.gguf",
    api_key="fake",
    api_base="http://localhost:1234/v1",
    is_chat_model=True
)
Settings.embed_model = LangchainEmbedding(GPT4AllEmbeddings(model='http://localhost:1234/v1/embeddings'))
whoosh = "/home/headquarters/Documents/Guardian/WebIndex/"

def QuestionCheck(collection_name, question):

    # Load from disk
    db2 = chromadb.PersistentClient(path="/home/headquarters/Documents/Guardian/VectorDatabase/")
    chroma_collection = db2.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    evaluator = QuestionEvaluator(vector_store)
    result = evaluator.evaluate_response(question)
    if result == True:
        response = handpicked_query(vector_store, question)
        response_content = response.response  # Accessing the actual response content
        return response_content
    else:
        return {
            "response" : result,
            "match" : False,
        }
    # Query

def handpicked_query(vector_store, question):
    corrective_rag= CorrectiveRAGPack(vector_store,Settings.llm,whoosh)
    response = corrective_rag.run(question, similarity_top_k=5)
    return response

def main():
    st.title("Guardian")
    collection_name = st.text_input("Enter the name of the collection to pull:")
    question = st.text_area("Write a query for the chatbot:")
    if st.button("Submit"):
        with st.spinner("Processing..."):
            check = QuestionCheck(collection_name, question)
            if isinstance(check, dict) and check.get("match") is False:
                check = check["response"]
                st.error("Question is too broad. Pick an operating system of choice to increase accuracy.")
                st.warning("Please re-enter your question with a specific operating system.")
                st.subheader("Metadata IDs:")
                seen_file_names = set()
                st.success(check['metadata_ids'])
                for idx, metadata in enumerate(check["metadata_ids"], start=1):
                    file_name = metadata.get('file_name', 'Unknown')
                    cleaned_file_name = re.sub(r'[^\w\s]', '', file_name)
                    if cleaned_file_name not in seen_file_names:
                        st.success(f"{idx}. {cleaned_file_name}")
                        seen_file_names.add(cleaned_file_name)
            elif isinstance(check, str) and check.strip() == 'Empty Response':
                st.success('Context Not Provided in Database')
            else:
                st.success(check)

if __name__ == "__main__":
    main()