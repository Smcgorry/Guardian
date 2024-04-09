from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.instructor import InstructorEmbedding
import chromadb

# Prompt user for input directory and vectorstore name
input_dir = input("Enter the directory path to load documents from: ")
collection_name = input("Enter the name of the collection to create: ")

# Define Embedding Model
embed_model = InstructorEmbedding(model_name="hkunlp/instructor-large", device='cuda')

documents = SimpleDirectoryReader(
    input_dir=input_dir
)

docs = documents.load_data()
print(f"Loaded {len(docs)} docs")

# Save to Database
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