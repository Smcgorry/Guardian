# Guardian
Offline CRAG Chatbot with updateable Database and user prompt evaluator. Used for querying and storing local documents. Has Customizable GUI with interchangeable LLM and Embedding Model. Built over ChromaDB, LLamaIndex, and LM Studio.

# Conda Environment SetUp:
1. conda create -n (venv name) python=>3.10

2. conda activate (venv name)
   
3. pip install -r requirements.txt

# LM Studio Instance SetUP:
LM Studio Instances: 

## For Windows 10/11: 
-Download LM studio from the link:  https://lmstudio.ai/	 

-To download the models search them up and download any model you want.

-After downloading the model head to the Local Inference Server

-The model you downloaded you can load and then start the server. 

## For Ubuntu:  
-Install the AppImage from https://lmstudio.ai/ and then run it through the terminal to spin up the LM studio UI. 
  
-To download the models search them up and download any model you want.

-After downloading the model head to the Local Inference Server

-The model you downloaded you can load and then start the server.

## LM Studio GPU SetUp:

-In the Local Inference Server select 'Cuda' under the 'GPU Offload' located right side Panel.

-Depending on hardware choose how many layers you want to load selecting '-1' will load max layers. Layers equate to VRAM on your card. 

### Debugging:
-Instance will crash if VRAM is higher than your hardware can handle.

-Reload Model to set 'GPU Offload' if changed after stopping server.  

# GUI QuickStart:
1. Change Directory to where ‘Guardian.py’ is stored 

2. Run Command: 

    >streamlit run Guardian.py

3. Navigate to the http://localhost:8501 to pull up the GUI

# DevScript:
- The Dev Script is used to upload documents to your local database with ChromaDB in a folder of your choosing. Storage used later to load documents off of with GUI. Choose Vectorstore name for storage and Directory path for folder with Documents to import into database. 
## Key Functions:
-Load: Can load documents of choosing to the data in the directory '/FunctionRecs/UploadData/' to a new VectorStore also labeling the MetaData ID of the group of Documents.
-Add: You can add data of choice in the directory '/FunctionRecs/AddData/' to an already existing VectorStore labeling MetaData ID of the document group. 

### Script:
'DevChrom.py'
```python
import os
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

#Define embedding function
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
```
# GUI Script SetUp:
The GUI script is customizable via what LLM you can use off LM Studio, where your database is stored locally to pull from, and type of vectorstore you want to pull. 

## Key Features:
- QuestionEvaluator: When the question is too broad it will reprompt the user to pick a topic based on the context of the question. If question matches a topic it won't reprompt and follow through the rest of the code. 
- If question is too broad pick type of data your question is related too. If question doesn't relate to data in vectorestore reponse 'Context not provided in Database'.

### Script: 

'Guardian.py'
```python
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
    db2 = chromadb.PersistentClient(path="/home/headquarters/Documents/Guardian/TestDB/")
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
```

# Offline Web Index:
Run 'Scraper.py' to add websites relating to Data in the Vectorstore with a keyword for the topic. Apart of CRAG(Corrective Retrieval Augmented Generation) which when an answer is incorrect or ambiguous(Data may or may not be related to content in the Database so it parses through offline index for context)
## Files Relating to Index:
-URls.json: All the URLs pulled from the OpenAI.
-AccessibleURLs.json: Akk the URLs that were successfully connected too. 
-HTML.json: All the html code parsed from the urls.




