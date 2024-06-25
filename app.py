import base64
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Any, Dict, Tuple

#useful libraries
from fastapi.responses import HTMLResponse
import requests
import sys
from icecream import ic
import pprint
from pypdf import PdfReader
import os
import pymupdf

from dotenv import load_dotenv

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
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Weaviate
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.document_loaders import PyMuPDFLoader

#For gemini
import google.generativeai as genai

#NLP libs
import nltk
import csv
from nltk.tokenize import sent_tokenize
import tempfile
import shutil
import pandas as pd
import spacy
from collections import Counter
import seaborn as sns
from matplotlib.figure import Figure
from io import BytesIO

#mongo
from pymongo import MongoClient

#pinecone
from langchain_pinecone import PineconeVectorStore

#custom
#from HaystackRetriever import HaystackRetriever

import firebase_admin
from firebase_admin.auth import verify_id_token

load_dotenv()



#Configure gemini api
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

MONGO_URI = os.getenv('MONGO_URI')
cluster = MongoClient(MONGO_URI)

DB_NAME = 'pdfs'
COLLECTION_NAME = 'pdfs_collection'
#NAMESPACE = 'pdfs.pdfs_collection'


MONGODB_COLLECTION = cluster[DB_NAME][COLLECTION_NAME]
vector_search_index = "vector_index"


PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

# Hugging Face model for embeddings.
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device': 'cpu'}
# model_kwargs = {'device': 'cuda'}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
)

'''
VECTOR_STORE = MongoDBAtlasVectorSearch.from_connection_string(
    connection_string = MONGO_URI,
    namespace = NAMESPACE,
    embedding = embeddings,
    index_name = vector_search_index
)
'''


INDEX_NAME = 'haystack'

VECTOR_STORE = PineconeVectorStore(
    pinecone_api_key=PINECONE_API_KEY,
    embedding = embeddings,
    index_name = INDEX_NAME,
)

#creating an instance of FastAPI
app = FastAPI()
origins = [
    "http://localhost:5173",
    "localhost:5173"
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Authentication
GOOGLE_APPLICATION_CREDENTIAL = os.getenv('GOOGLE_APPLICATION_CREDENTIAL')
firebase_admin.initialize_app()

# SECURITY/AUTHENTICATION
# function verify token

def get_firebase_user_from_token(token):
   
    try:
        if not token:
            # raise and catch to return 401, only needed because fastapi returns 403
            # by default instead of 401 so we set auto_error to False
            raise ValueError("No token")
        user = verify_id_token(token.credentials)
        return user
    # lots of possible exceptions, see firebase_admin.auth,
    # but most of the time it is a credentials issue
    except Exception:
        # we also set the header
        # see https://fastapi.tiangolo.com/tutorial/security/simple-oauth2/
        print("Invalid")
        return None

#################################################### SUBMITING PDFS ####################################################
def submit_docs_for_rag(submitted_pdf, pdf_directory):
    directory_path = pdf_directory
    pdf = submitted_pdf

    # Load the pdf.
    loaders = []
    loader = PyMuPDFLoader(directory_path + "/" + pdf)
    loaders.append(loader)

    print("len(loaders) =", len(loaders))

    data = loader.load()

    print("len(data) =", len(data), "\n")

    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    docs = text_splitter.split_documents(data)#, metadatas = [{'doc_id' : 1}])
    doc_content = [doc.page_content for doc in docs]
    print(doc_content)

    # Print the document.
 #   print("\n\n\n")
 #   pprint.pprint(docs)

    return doc_content
########################################################################################################################

######################################################### NLPS #########################################################

NLP = spacy.load("en_core_web_sm")
#nltk.download('punkt')

async def search_keyword_in_pdfs(pdf_files: List[UploadFile], keyword: str) -> Tuple[List[Dict[str, Any]], int]:
    """
    Searches for a keyword in a list of PDF files and returns the keyword counts and total count.

    Args:
        pdf_files (List[UploadFile]): A list of UploadFile objects representing the PDF files to search.
        keyword (str): The keyword to search for.

    Returns:
        Tuple[List[Dict[str, Any]], int]: A tuple containing the keyword counts per file and the total count.
            - keyword_counts (List[Dict[str, Any]]): A list of dictionaries containing the keyword count, metadata, and page number for each file.
            - total (int): The total count of the keyword across all files.
    """
    
    keyword_counts = [] # List to store keyword counts per file
    total = 0  

    for pdf_file in pdf_files:
        
        reader = PdfReader(pdf_file.file)
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
                    metadata = {'filename': pdf_file.filename, 'page': page_num, 'Sources': context_sent}
                    count_dict = {'keyword': keyword, 'count': keyword_count, 'metadata': metadata.copy()}
                    keyword_counts.append(count_dict)
                    total += keyword_count

            page_num += 1  # Increase page number

    return keyword_counts, total

async def concepts_frequencies_in_pdfs(pdf_files: List[UploadFile]):
    
    concept_counter = Counter()

    for pdf_file in pdf_files:
        # Sample text
        text = ''

        reader = PdfReader(pdf_file.file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:  # Checking if text extraction is successful
                text += page_text.lower()

        # Process the text with spaCy
        doc = NLP(text)

        # Extract nouns (or other POS you're interested in)
        concepts = [token.text for token in doc if token.pos_ == "NOUN"]

        # Count the frequency of each concept
        concept_counter.update(concepts)
    
    #filtered_concepts = {concept: freq for concept, freq in concept_counter.items() if freq > 5}
    filtered_concepts = {concept: freq for concept, freq in concept_counter.most_common(20)}
    sorted_concept_freq = dict(sorted(filtered_concepts.items(), key=lambda item: item[1], reverse=True))
    #print(sorted_concept_freq)
    num_concepts = len(sorted_concept_freq)
    fig_width = max(10, num_concepts * 0.5)

    # Create a data frame from the sorted concept frequencies
    concept_df = pd.DataFrame(list(sorted_concept_freq.items()), columns=['Concept', 'Frequency'])
    buf = BytesIO()

    # Create the bar plot
    fig = Figure(figsize=(fig_width, 6))
    ax = fig.add_subplot(111)
    #plt.figure(figsize=(fig_width, 6))
    sns.barplot(x='Concept', y='Frequency', data=concept_df, palette='magma', ax=ax, hue='Concept', legend=False)

    # Add labels and title
    ax.set_xlabel('Concepts')
    ax.set_ylabel('Frequency')
    ax.set_title('Frequency of Concepts in Text (Top 20)')
    ax.tick_params(axis='x', labelrotation=45)  # Rotate the x-axis labels

    # Save the plot
    fig.tight_layout()
    fig.savefig(buf, format='png', bbox_inches='tight') #CHANGE
    
    data = base64.b64encode(buf.getbuffer()).decode("ascii")

    return {"status": "success", "message": "Concepts extracted successfully.", "image_data": data}


########################################################################################################################

#################################################### RAG QA 1 TO 1 #####################################################
def add_docs(chosen_pdf, pdf_directory, namespace, doc_id):
    pdf = submit_docs_for_rag(chosen_pdf, pdf_directory)

    # Create Weaviate vector store (database).
#    client = weaviate.Client(
#    embedded_options = EmbeddedOptions()
#    )
    print(f"Creating vector store for {len(pdf)} chunks")
        
    VECTOR_STORE.add_texts(
        texts = pdf,
        namespace = namespace,
        metadatas = [{'doc_id' : doc_id} for i in pdf]
    )


def Haystack_qa_1(chosen_pdf, pdf_directory, query: str, namespace, doc_id):

    #add_docs(chosen_pdf, pdf_directory)
    
    RETRIEVER_STORE = PineconeVectorStore.from_existing_index(
        embedding = embeddings,
        index_name = INDEX_NAME,
        namespace = namespace
    )

    retriever = RETRIEVER_STORE.as_retriever(
        search_type = "similarity_score_threshold", 
        search_kwargs = {"k": 10, 'score_threshold': 0.7, 'filter': {"doc_id": doc_id}}   
    )

    # Retrieve documents
    print(f"Retrieved documents for PDF:")
    pprint.pprint(retriever)

    # Check if retrieved_docs is empty
    if not retriever:
        print(f"No relevant documents found for PDF")
    

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
    generation_config = {"temperature": 0.9, # Increasing the temperature, the model becomes more creative and takes longer for inference.
    "top_p": 1, "top_k": 1, "max_output_tokens": 2048}

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", generation_config=generation_config,
                                safety_settings={
                                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
                                })
    # RAG chain
    rag_chain = (
        {"context": retriever,  "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        # | hf
        | StrOutputParser()
    )
    reply = rag_chain.invoke(query)
    print("RAG REPLY:" + reply)
    return reply
########################################################################################################################

################################################## RAG QA 1 TO MANY ####################################################
def Haystack_qa_many(pdf_directory, query: str):
    pdfs = []
    replies = []
    for file in os.listdir(pdf_directory):
        if file.endswith(".pdf"):
            #Separately chunks all pdfs individually so that they can be embedded individually
            pdfs.append(submit_docs_for_rag(file, pdf_directory)) 
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
    for i, pdf in enumerate(pdfs):
        print(f"Creating vector store for {len(pdf)} chunks")
        # Initialize the Weaviate vector search with the document segments.
        # Create a vector store (database) named vector_search from the sample documents.
        vector_search = Weaviate.from_documents(
            client = client,
            documents = pdf,
            embedding = embeddings,
            by_text = False
        )

        # Vector Search retreiver
        retriever = vector_search.as_retriever(
            search_type = "similarity", 
            search_kwargs = {"k": 10, "score_threshold": 0.89}
        )

        # Retrieve documents
        print(f"Retrieved documents for PDF {i + 1}:")
        pprint.pprint(retriever)

        # Check if retrieved_docs is empty
        if not retriever:
            print(f"No relevant documents found for PDF {i + 1}")

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
        generation_config = {"temperature": 0.9, # Increasing the temperature, the model becomes more creative and takes longer for inference.
        "top_p": 1, "top_k": 1, "max_output_tokens": 2048}

        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", generation_config=generation_config,
                                    safety_settings={
                                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
                                    })
        # RAG chain
        rag_chain = (
            {"context": retriever,  "question": RunnablePassthrough()}
            | custom_rag_prompt
            | llm
            # | hf
            | StrOutputParser()
        )
        reply = rag_chain.invoke(query)
        print("RAG REPLY:" + reply)
        replies.append(reply)
    return replies
########################################################################################################################

# 1 to 1 and 1 to many
# 1 query for 1 doc and 1 query for multiple docs separately
# Grab all the data from the NLP and put them into a csv file
# Topic modeling important
# Allow users to choose chunk sizes maybe?

#defining the routes

#route to the home page
@app.get("/",response_class=HTMLResponse)
def index():
    """
    This function handles the index route of the application.
    
    Returns:
        dict: A dictionary containing a welcome message.
    """
    return """
    <html>
    <head>
        <title>Upload PDFs</title>
    </head>
    <body>
        <h1>Upload PDFs</h1>
        <form action="/conceptsfrequencies/" method="post" enctype="multipart/form-data">
            <input type="file" name="files" accept="application/pdf" multiple>
            <button type="submit">Upload</button>
        </form>
    </body>
    </html>
    """

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
    Extracts text from PDF files and saves them as text files.

    Args:
        files (list[UploadFile]): A list of UploadFile objects representing the PDF files to process.

    Returns:
        dict: A dictionary containing the status and message of the extraction process.
            - If the extraction is successful, the status will be "success" and the message will be "Text extracted successfully."
            - If an error occurs during the extraction, the status will be "fail" and the message will contain the error details.
    """
    try:
        for file in files:
            reader = PdfReader(file.file)
            text = ""
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text()
            # Save the extracted text to a .txt file
            # For now it is saved in the "texts" directory
            # Eventually, we will need to store it in a database or file system
            output_filename = "texts/" + file.filename.replace(".pdf", ".txt")
            with open(output_filename, "w", encoding="utf-8") as txt_file:
                txt_file.write(text)
        return {"status": "success", "message": "Text extracted successfully."}
    except Exception as e:
        return {"status": "fail", "message": f"Failed to extract text. Error ocurred: {e}"}

@app.post("/searchkeyword/")
async def search_keyword(files: List[UploadFile], keyword: str) -> Dict[str, Any]:
    """
    Searches for a keyword in a list of uploaded files.

    Args:
        files (List[UploadFile]): A list of uploaded files to search in.
        keyword (str): The keyword to search for.

    Returns:
        Dict[str, Any]: A dictionary containing the search results.
            - "status": The status of the search ("success" or "fail").
            - "keyword": The keyword that was searched for.
            - "total": The total number of occurrences of the keyword in the files.
            - "results": A dictionary containing the keyword counts for each file.

    Raises:
        Exception: If an error occurs during the search.

    """
    try:
        keyword_counts, total = await search_keyword_in_pdfs(files, keyword)
        return {"status":"success", "keyword": keyword, "total": total, "results": keyword_counts}
    except Exception as e:
        return {"status":"fail", "keyword": keyword, "message": f"Failed to search for keyword. Error ocurred: {e}"}

@app.post("/conceptsfrequencies/")
async def concept_frequencies(files: List[UploadFile]):

    try:
        return await concepts_frequencies_in_pdfs(files)
    except Exception as e:
        return {"status": "fail", "message": f"Failed to extract concepts. Error ocurred: {e}"}
    


#VECTOR_STORE.delete(delete_all=True, namespace="user_1")
#add_docs("OWASP Application Security Verification Standard 4.0.3-en.pdf", "pdfs", 'user_1', 1)
#add_docs("file_87.pdf", "pdfs", 'user_1', 2)
#Haystack_qa_1("a", "b", "What is authorization?", 'user_1', 1)

#submit_docs_for_rag("OWASP Application Security Verification Standard 4.0.3-en.pdf", "pdfs")
#MONGODB_COLLECTION.delete_many({})
