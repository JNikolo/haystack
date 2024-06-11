from fastapi import FastAPI

#useful libraries
import requests
import sys
from icecream import ic
import pprint
#from PyPDF2 import PdfReader
import os
import pymupdf

#Embeddings temp
import weaviate
from weaviate.embedded import EmbeddedOptions

#Langchain for RAG
from langchain.callbacks.tracers import ConsoleCallbackHandler
#from langchain.chat_models import ChatOpenAI
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,

)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Weaviate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.document_loaders import PyMuPDFLoader

#For gemini
import google.generativeai as genai

#Configure gemini api
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

#creating an instance of FastAPI
app = FastAPI()

#################################################### SUBMITING PDFS ####################################################
def sumbit_docs():
    directory_path = "/pdfs" ########## Directory that contains the pdf docs #############

    pdfs = []
    for file in os.listdir(directory_path):
        if file.endswith(".pdf"):
            pdfs.append(file)
    # Load the TXT.
    loaders = []
    for pdf in pdfs:
        loader = PyMuPDFLoader(directory_path + "/" + pdf)
        loaders.append(loader)

    print("len(loaders) =", len(loaders))

    data = []
    for loader in loaders:
        data.append(loader.load())

    print("len(data) =", len(data), "\n")

    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    docs = []
    for doc in data:
        chunk = text_splitter.split_documents(doc)
        docs.append(chunk)

    # Debugging purposes.
    # Print the number of total documents to be stored in the vector database.
    total = 0
    for i in range(len(docs)):
        if i == len(docs) - 1:
            print(len(docs[i]), end="")
        else:
            print(len(docs[i]), "+ " ,end="")
        total += len(docs[i])
    print(" =", total, " total documents\n")

    # Print the first document.
    pprint.pprint(docs[0])
    print("\n\n\n")

    # Print the total number of PDF files.
    # docs is a list of lists where each list stores all the documents for one PDF file.
    print(len(docs))

    # Merge the documents to be embededed and store them in the vector database.
    merged_documents = []

    for doc in docs:
        merged_documents.extend(doc)

    # Print the merged list of all the documents.
    print("len(merged_documents) =", len(merged_documents))
    pprint.pprint(merged_documents)

    return merged_documents

########################################################################################################################

####################################################### RAG QA #########################################################
def Haystack_qa(merged_documents, query):
    # Hugging Face model for embeddings.
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    # model_kwargs = {'device': 'cuda'}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
    )

    # Create Weaviate vector store (database).
    client = weaviate.Client(
    embedded_options = EmbeddedOptions()
    )

    # Initialize the Weaviate vector search with the document segments.
    # Create a vector store (database) named vector_search from the sample documents.
    vector_search = Weaviate.from_documents(
        client = client,
        documents = merged_documents,
        embedding = embeddings,
        by_text = False
    )

    # Vector Search retreiver
    retriever = vector_search.as_retriever(
        search_type = "similarity", 
        search_kwargs = {"k": 10, "score_threshold": 0.89}
    )

    # Define a prompt template.
    # LangChain passes these documents to the {context} input variable and the user's query to the {question} variable.
    template = """
    You are looking for the specified keywords and answering questions based on the keywords.
    Use the following pieces of retrieved context to answer the question at the end.
    If you don't know the answer, just say that you don't know.

    Context: {context}

    Question: {question}
    """
    custom_rag_prompt = ChatPromptTemplate.from_template(template)

    # Model settings
    generation_config = {
    "temperature": 0.9, # Increasing the temperature, the model becomes more creative and takes longer for inference.
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
    }

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                                generation_config=generation_config,
                                safety_settings={
                                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
                                }
                                )
    # RAG chain
    rag_chain = (
        {"context": retriever,  "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        # | hf
        | StrOutputParser()
    )
    return rag_chain.invoke(query)
########################################################################################################################

#defining the routes

#route to the home page
@app.get("/")
def index():
    """
    This function handles the index route of the application.
    
    Returns:
        dict: A dictionary containing a welcome message.
    """
    return {"Welcome": "To Haystack"}

#route to the /login page
@app.post("/login")
def login(username: str, password: str):
    """
    Logs in the user with the provided username and password.

    Args:
        username (str): The username of the user.
        password (str): The password of the user.

    Returns:
        dict: A dictionary containing the username and password.

    """
    return {"username": username, "password": password}

#route to the /querydocuments page
@app.get("/querydocuments/{query}/topk/{top_k}")
def read_item(query: str, top_k: int):
    """
    Reads an item with the given query and top_k parameters.

    Args:
        query (str): The query string.
        top_k (int): The number of top items to retrieve.

    Returns:
        dict: A dictionary containing the query and top_k parameters.
    """
    return {"q": query, "top_k": top_k}