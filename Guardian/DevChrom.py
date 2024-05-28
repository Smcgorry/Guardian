import os
#ID for Docs in DB
import uuid
#VectorStore Packages:
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
#Required Model(AI & Embedding) Packages:
#from llama_index.embeddings.instructor import InstructorEmbedding
from Packages.customclass import LlamaIndexEmbeddingAdapter
from llama_index.llms.openai_like import OpenAILike
from langchain_community.embeddings import GPT4AllEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding

#Node Parsing Packages:
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode
from llama_index.core.node_parser import SimpleFileNodeParser

#LM Studio Model
Settings.llm = OpenAILike(
    model="model-id",
    api_key="fake",
    api_base="http://localhost:1234/v1",
    is_chat_model=True
)

# Define embedding function
Settings.embed_model = LangchainEmbedding(GPT4AllEmbeddings(model='http://localhost:1234/v1/embeddings'))

#Path for Database Store
db = chromadb.PersistentClient(path="../VectorDatabase/")

#Function for Loading new info into Database
def loaddata(collection_name, nodes):
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    nodes = [node for node in nodes if node.get_content(metadata_mode=MetadataMode.EMBED) != ""]
    index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=Settings.embed_model)
    print(f"Data loaded into collection '{collection_name}'.")

#Function for Adding into Database
def add_data(collection_name, nodes, input_directory):
    chroma_collection = db.get_collection(collection_name,embedding_function=LlamaIndexEmbeddingAdapter(Settings.embed_model))
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    for node in nodes:
        node_embedding = Settings.embed_model.get_text_embedding(node.get_content(metadata_mode="all"))
        node.embedding = node_embedding
    nodes = [node for node in nodes if node.get_content(metadata_mode=MetadataMode.EMBED) != "" and node.get_embedding() is not None]
    vector_store.add(nodes=nodes)

def list_directories(base_path):
    # List all directories in the base path
    directories = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    # Display the available directories
    if directories:
        print("Available directories:")
        for directory in directories:
            print(directory)
    else:
        print("No directories found in the base path.")
    return directories

if __name__ == '__main__':
    #User Inputs:
    choice = input("Do you want to 'load' new data or 'add' to an existing collection? (load/add): ").strip().lower()
    collection_name = input("Enter the name of the collection: ")
    #Node Parser:
    Settings.node_parser = SentenceSplitter(chunk_size=500, chunk_overlap=20)

    #Document Loader
    if choice == 'load':
        input_dir = '../FunctionRecs/DataUpload/'
        metadata_name = str(input('Please provide a title for this group of documents to help you easily identify and recall them later: '))
        filename_fn = lambda filename: {"file_name": metadata_name}
        documents = SimpleDirectoryReader(input_dir=input_dir,file_metadata=filename_fn)
        docs = documents.load_data()
        print(f"Loaded {len(docs)} docs")
    #Text to Node
        nodes = Settings.node_parser.get_nodes_from_documents(docs)
        loaddata(collection_name, nodes)
    #Document Add to Store
    elif choice == 'add':
        basepath = '../FunctionRecs/AddData/'
        available_directories = list_directories(basepath)
        if available_directories:
            while True:
                input_directory = input(f'Please enter the directory name from the above list: ').strip()
                if input_directory in available_directories:
                    break
                else:
                    print(f'Invalid directory name. Please choose a directory from the list.')
        else:
            print("No directories available to choose from.")
        input_dir = f'{basepath}{input_directory}'
        print(input_dir)
        metadata = lambda filename: {"file_name": input_directory}
        documents = SimpleDirectoryReader(input_dir=input_dir,file_metadata=metadata)
        docs = documents.load_data()
        for doc in docs:
            if "page_label" in doc.metadata:
                del doc.metadata["page_label"]
        for doc in docs:
            print(doc.metadata)
        print(f"Loaded {len(docs)} docs")
        nodes = Settings.node_parser.get_nodes_from_documents(docs)
        # Here you need to ensure docs is a list of (document, embedding) tuples
        # This is a placeholder; you need to generate embeddings for each doc
        add_data(collection_name, nodes, input_directory)
    else:
        print("Invalid choice. Please enter 'load' or 'add'")
