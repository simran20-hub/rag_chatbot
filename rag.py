from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
from huggingface_hub import login

load_dotenv()

CHUNK_SIZE = 1000
EMBEDDING_MODEL = "Alibaba-NLP/gte-base-en-v1.5"
VECTOR_STORE_PATH = Path(__file__).parent/"resources"/"vector_store"
COLLECTION_NAME = "real_estate"
hf_token = os.getenv("hugging_api_key")

# Ensure token is set
if not hf_token:
    raise ValueError("HUGGINGFACE API token not set. Set HuggingFace API token in .env file")

# Authenticate before using the model
login(token=hf_token)

llm = None
vector_store = None

def initialize_components():
    global llm, vector_store
    if llm is None:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.9, max_tokens=5000)

    if vector_store is None:
        ef = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True}
        )

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=ef,
            persist_directory=str(VECTOR_STORE_PATH)
        )

def process_urls(urls):
    """
    This function scraps data from the URLs and stores it in a vector db.
   
    :param urls: Description
    :return
    """

    #print("Initialize components")
    yield "Initializing components..."
    initialize_components()
    vector_store.reset_collection()

    #print("Load Data")
    yield "Resetting vector store..."
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    #print(Split Text")
    yield "Splitting text into chunks..."
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=CHUNK_SIZE,
    )

    docs = text_splitter.split_documents(data)

    # **ADD THESE CHECKS**
    # Ensure docs is a list, not a set
    if isinstance(docs, set):
        docs = list(docs)
    
    # Ensure we have documents
    if not docs:
        yield "No documents found to process."
        return


    #print("Add docs to vector db")
    yield "Adding chunks to vector store..."
    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(docs, ids=uuids)

    yield "Done adding docs to vector database."

def generate_answer(query):
    if not vector_store:
        raise RuntimeError("Vector Database is not initialized")
    chain=RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vector_store.as_retriever())

    result=chain.invoke({"question":query}, return_only_outputs=True)

    sources=result.get("sources"," ")

    return result['answer'], sources
