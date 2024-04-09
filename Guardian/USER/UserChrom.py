from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from IPython.core.display import Markdown
from IPython.core.display_functions import display
from InstructorEmbedding import INSTRUCTOR
import chromadb

from llama_index.llms.azure_openai import AzureOpenAI

llm = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
embed_model = INSTRUCTOR(model_name="hkunlp/instructor-base")

collection_name = input("Enter the name of the collection to create: ")
question = input("Write a query for the chatbot: ")

# load from disk
db2 = chromadb.PersistentClient(path="./chroma_db")

chroma_collection = db2.get_or_create_collection(collection_name)

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

service_context = ServiceContext.from_defaults(llm=llm)

index = VectorStoreIndex.from_vector_store(
    vector_store,
    service_context=service_context,
)

# Query Data
query_engine = index.as_query_engine()

response = query_engine.query(question)

display(Markdown(f"{response}"))