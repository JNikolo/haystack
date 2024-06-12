from fastapi import FastAPI, UploadFile

#useful libraries
import requests
import sys
from icecream import ic
import pprint
from pypdf import PdfReader
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

#NLP libs
import nltk
import csv
from PyPDF2 import PdfReader
from nltk.tokenize import sent_tokenize
import requests
import tempfile
import shutil
import pandas as pd

#Configure gemini api
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

#creating an instance of FastAPI
app = FastAPI()

#################################################### SUBMITING PDFS ####################################################
def submit_docs_for_rag():
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

######################################################### NLPS #########################################################
def search_keyword_in_pdfs(pdf_files, keyword):
    keyword_counts = []  # List to store keyword counts per file
    total = 0

    for pdf_file in pdf_files:
        with open(pdf_file, "rb") as pdf:
            reader = PdfReader(pdf)
            page_num = 1  # Variable to keep track of the page number

            for page in reader.pages:
                page_text = page.extract_text()

                if page_text:  # Checking if text extraction is successful
                    keyword_count = 0
                    sentences = sent_tokenize(page_text)
                    context_sent = []

                    for j, sentence in enumerate(sentences):
                        sentence_lower = sentence.lower()
                        if keyword.lower() in sentence_lower:
                            keyword_count += sentence_lower.count(keyword.lower())

                            prev_sentence = sentences[j - 1] if j > 0 else ''
                            next_sentence = sentences[j + 1] if j < len(sentences) - 1 else ''
                            context = f"{prev_sentence} {sentence} {next_sentence}".strip()
                            context_sent.append(context)

                    if keyword_count > 0:
                        # Store metadata and keyword count
                        metadata = {'filename': pdf_file, 'page': page_num, 'Sources': context_sent}
                        count_dict = {'keyword': keyword, 'count': keyword_count, 'metadata': metadata.copy()}
                        keyword_counts.append(count_dict)
                        count = count_dict['count']
                        total += count

                page_num += 1  # Increase page number



    print("Keyword total count:", total)
    return keyword_counts
########################################################################################################################

####################################################### RAG QA #########################################################
def Haystack_qa(merged_documents, query: str):
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

# 1 to 1 and 1 to many
# 1 query for 1 doc and 1 query for multiple docs separately
# Grab all the data from the NLP and put them into a csv file
# Topic modeling important
# Allow users to choose chunk sizes maybe?

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

@app.post("/uploadfiles/")
async def create_upload_files(files: list[UploadFile]):
    """
    Extracts text from the first page of a PDF file. Allows multiple files to be uploaded (not at once)

    Args:
        files (list[UploadFile]): A list of UploadFile objects representing the uploaded files.

    Returns:
        dict: A dictionary containing the status and extracted text.
            - If the PDF file is successfully processed, the status will be "success" and the extracted text will be returned.
            - If the PDF file is empty or cannot be processed, the status will be "wump wump" and a default text will be returned.
    """
    doc = PdfReader(files[0].file)
    if doc:
        return {"status": "success", "text": doc.pages[0].extract_text()}
    else:
        return {"status": "wump wump", "text": "empty"}

