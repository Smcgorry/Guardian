# Guardian
Offline RAG Chatbot with updateable Database. Used for querying and storing local documents. Has Customizable GUI with interchangeable LLM and Embedding Model. Built over ChromaDB and LLamaIndex.

# Conda Environment SetUp:
1. conda create -n (venv name) python=3.10

2. conda activate (venv name)
   
3. pip install -r requirements.txt

# GUI QuickStart:
1. Change Directory to where ‘Guardian.py’ is stored 

2. Run Command: 

    >streamlit run Guardian.py

3. Navigate to the http://localhost:8501 to pull up the GUI

# DevScript:
The Dev Script is used to upload documents to your local database with ChromaDB in a folder of your choosing. Storage used later to load documents off of with GUI. Choose Vectorstore name for storage and Directory path for folder with Documents to import into database.

'''python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from InstructorEmbedding.instructor import INSTRUCTOR
from llama_index.readers.file import DocxReader
import chromadb

## Prompt user for input directory and vectorstore name
input_dir = input("Enter the directory path to load documents from: ")
collection_name = input("Enter the name of the collection to create: ")

## Define Embedding Model
embed_model = INSTRUCTOR("hkunlp/instructor-base")

documents = SimpleDirectoryReader(
    input_dir=input_dir
)

docs = documents.load_data()
print(f"Loaded {len(docs)} docs")

## Save to Database
#Pick folder for Storage
db = chromadb.PersistentClient(path="./chroma_db")
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
'''
