# Guardian
Offline RAG Chatbot with updateable Database. Used for querying and storing local documents. Has Customizable GUI with interchangeable LLM and Embedding Model. Built over ChromaDB and LLamaIndex.

# Conda Environment SetUp:
1. conda create -n (venv name) python=3.10

2. conda activate (venv name)
   
3. pip install -r requirements.txt

# GUI QuickStart:
1. Change Directory to where ‘GuardianGUI.py’ is stored 

2. Run Command: 

    >streamlit run Guardian.py

3. Navigate to the http://localhost:8501 to pull up the GUI

# DevScript:
The Dev Script is used to upload documents to your local database with ChromaDB in a folder of your choosing. Storage used later to load documents off of with GUI. Choose Vectorstore name for storage and Directory path for folder with Documents to import into database.

'GuardianDev.py'
```python
# Dependencies
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.instructor import InstructorEmbedding
import chromadb

# Prompt user for input directory and vectorstore name
input_dir = input("Enter the directory path to load documents from: ")
collection_name = input("Enter the name of the collection to create: ")

# Define Embedding Model
embed_model = InstructorEmbedding(model_name="hkunlp/instructor-base", device='cuda')

documents = SimpleDirectoryReader(
    input_dir=input_dir
)

docs = documents.load_data()
print(f"Loaded {len(docs)} docs")

# Save to Database
# Pick folder for Storage
db = chromadb.PersistentClient(path="Where-your-database-is-located")
#VectorStore to Create
chroma_collection = db.get_or_create_collection(collection_name)

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

service_context = ServiceContext.from_defaults(embed_model=embed_model,llm=None,
                                               chunk_size=800,
                                               chunk_overlap=20)

index = VectorStoreIndex.from_documents(
    docs, storage_context=storage_context, service_context=service_context
)
```
# GUI Script SetUp:
The GUI script is customizable via what LLM you can use off LM Studio, where your database is stored locally to pull from, and type of vectorstore you want to pull.

'GuardianGUI.py'
```python
# Dependencies
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
```
