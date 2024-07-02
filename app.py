from fastapi import FastAPI, UploadFile, Form, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Any, Dict, Tuple

#useful libraries
from fastapi.responses import HTMLResponse
from icecream import ic
import pprint
from pypdf import PdfReader
import os

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
from nltk.tokenize import sent_tokenize
import spacy
from collections import Counter, defaultdict
import re
from spacy.lang.en import English
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.parsing.preprocessing import preprocess_string
from gensim.corpora import Dictionary
from gensim.models import LdaModel

#pinecone
from langchain_pinecone import PineconeVectorStore

import firebase_admin
from firebase_admin.auth import verify_id_token

load_dotenv()

#Configure gemini api
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

DB_NAME = 'pdfs'
COLLECTION_NAME = 'pdfs_collection'
#NAMESPACE = 'pdfs.pdfs_collection'

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


""" def login():
    data = request.get_json()
    id_token = data['id_token']
    try:
        decoded_token = auth.verify_id_token(id_token)
        uid = decoded_token['uid']
        return jsonify({"message": "User authenticated successfully", "uid": uid}), 200
    except Exception as e:
        return jsonify({"message": str(e)}), 400 """

#################################################### SUBMITING PDFS ####################################################
def submit_docs_for_rag(submitted_pdf:UploadFile):
    #directory_path = pdf_directory
    #pdf = submitted_pdf

    reader = PdfReader(submitted_pdf.file)
    # Load the pdf.
    #loaders = []
    #loader = PyMuPDFLoader(directory_path + "/" + pdf)
    #loaders.append(loader)

    #print("len(loaders) =", len(loaders))

    #data = loader.load()

    #print("len(data) =", len(data), "\n")

    # Initialize the text splitter
    raw_text = ''
    for page in reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    docs = text_splitter.split_text(raw_text)#, metadatas = [{'doc_id' : 1}])
    #doc_content = [doc.page_content for doc in docs]
    #print(doc_content)

    # Print the document.
 #   print("\n\n\n")
 #   pprint.pprint(docs)

    return docs
########################################################################################################################

######################################################### NLPS #########################################################

NLP = spacy.load("en_core_web_sm")
PARSER = English()
nltk.download('stopwords')
EN_STOP = set(nltk.corpus.stopwords.words('english'))
#nltk.download('stopwords')
nltk.download('wordnet')

async def get_list_raw_text(pdf_files: List[UploadFile]) -> List[str]:
    all_pdf_text = []
    for pdf_file in pdf_files:
        pdf_reader = PdfReader(pdf_file.file)
        pdf_text = ' '.join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
        all_pdf_text.append(pdf_text)
    
    return all_pdf_text

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
    entity_info = defaultdict(lambda: {'frequency': 0, 'labels': set()})
    
    all_pdf_text = await get_list_raw_text(pdf_files)
                
    combined_text = '\n\n'.join(all_pdf_text)

        # Process the text with spaCy
    doc = NLP(combined_text)

    for ent in doc.ents:
        if ent.label_ in ['ORG', 'PERSON', 'GPE', 'PRODUCT', 'EVENT']:
            cleaned_text = re.sub('[^A-Za-z0-9]+', ' ', ent.text).strip()
            if cleaned_text:  # Check if cleaned_text is not empty
                entity_info[cleaned_text]['frequency'] += 1
                entity_info[cleaned_text]['labels'].add(ent.label_)

    sorted_entities = sorted(entity_info.items(), key=lambda x: x[1]['frequency'], reverse=True)[:10]
    result = []
    for entity, info in sorted_entities:
        result.append({'text': entity, 'frequency': info['frequency'], 'labels': list(info['labels'])})


    return result

async def tokenize(text):
    parser = English()
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens

async def get_lemma(word):
    lemma = wordnet.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma

async def prepare_text_for_lda(text):
    tokens = await tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in EN_STOP]
    tokens = [await get_lemma(token) for token in tokens]
    return tokens

async def topic_modeling_from_pdfs(pdf_files: List[UploadFile]):
    all_pdf_text = await get_list_raw_text(pdf_files)

    # Preprocess text
    preprocessed_documents = [await prepare_text_for_lda(line) for line in all_pdf_text]

    # Join tokens for TF-IDF vectorization
    processed_docs = [' '.join(doc) for doc in preprocessed_documents]

    # Apply TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_docs)
    feature_names = vectorizer.get_feature_names_out()

    # Create a mapping of words to their maximum TF-IDF scores
    word_to_max_score = {}
    for doc in range(tfidf_matrix.shape[0]):
        feature_index = tfidf_matrix[doc, :].nonzero()[1]
        for i in feature_index:
            word = feature_names[i]
            score = tfidf_matrix[doc, i]
            if word not in word_to_max_score or score > word_to_max_score[word]:
                word_to_max_score[word] = score

    # Get the top 50% threshold score
    scores = list(word_to_max_score.values())
    scores.sort(reverse=True)
    top_50_percent_threshold = scores[len(scores) // 2]

    # Filter words in each document to include only those above the threshold
    top_50_percent_words = []
    for doc in preprocessed_documents:
        filtered_words = [word for word in doc if word in word_to_max_score and word_to_max_score[word] > top_50_percent_threshold]
        top_50_percent_words.append(filtered_words)

    # Use 'top_50_percent_words' for further analysis or modeling

    text_data = top_50_percent_words

    # Create a dictionary and a document-term matrix
    dictionary = Dictionary(text_data)
    corpus = [dictionary.doc2bow(text) for text in text_data]

    # Build the LDA model
    num_topics = 5  # You can adjust this based on your dataset
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15, random_state=100)

    # Prepare output DataFrames
    topics = lda_model.show_topics(formatted=False)
    counter = Counter([word for tokens in preprocessed_documents for word in tokens])

    out_count = []
    out_importance = []

    for i, topic in enumerate(topics):
        for word, weight in topic[1]:
            out_count.append([word, i, counter[word]])
            out_importance.append([word, i, weight])
    
    df_count = pd.DataFrame(out_count, columns=['word', 'topic_id', 'word_count'])
    df_importance = pd.DataFrame(out_importance, columns=['word', 'topic_id', 'importance'])

    return df_count, df_importance

########################################################################################################################

#################################################### RAG QA 1 TO 1 #####################################################
def add_docs(chosen_pdf:UploadFile, namespace:str, doc_id:int):
    pdf = submit_docs_for_rag(chosen_pdf)

    # Create Weaviate vector store (database).
#    client = weaviate.Client(
#    embedded_options = EmbeddedOptions()
#    )
    print(f"Creating vector store for {len(pdf)} chunks")
    

    # Checking if pdf already exists in the database???????


    VECTOR_STORE.add_texts(
        texts = pdf,
        namespace = namespace,
        metadatas = [{'doc_id' : doc_id} for _ in range(len(pdf))]
    )


def Haystack_qa_1(query: str, namespace:str, doc_id:int):

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
def Haystack_qa_many(pdf_list, query: str):
    pdfs = []
    replies = []
    for file in pdf_list:
        #Separately chunks all pdfs individually so that they can be embedded individually
        pdfs.append(submit_docs_for_rag(file)) 
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
    
########################################################################################################################
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
        results = await concepts_frequencies_in_pdfs(files)
        return {"status": "success", "message": "Concept frequencies extracted successfully.", "results": results}
    except Exception as e:
        return {"status": "fail", "message": f"Failed to extract concepts. Error ocurred: {e}"}


@app.post("/topicmodeling/")
async def topic_modeling(files: List[UploadFile]):
    try:
        df_count, df_importance = await topic_modeling_from_pdfs(files)
        count = df_count.to_dict(orient='records')
        importance = df_importance.to_dict(orient='records')
        
        return {"status": "success", "message": "Topic modeling performed successfully.", "count": count, "importance": importance}
    except Exception as e:
        return {"status": "fail", "message": f"Failed to perform topic modeling. Error ocurred: {e}"}

########################################################################################################################
from pydantic import BaseModel
@app.delete("/delete_embeddings/")
def delete_embeddings(user_id: str):
    """
    Deletes the embeddings for the specified user.

    Args:
        user_id (str): The user ID for which to delete the embeddings.

    Returns:
        dict: A dictionary containing the status of the deletion process.
    """
    try:
        VECTOR_STORE.delete(delete_all=True, namespace=user_id)
        return {"status": "success", "message": "Embeddings deleted successfully."}
    except Exception as e:
        return {"status": "fail", "message": f"Failed to delete embeddings. Error ocurred: {e}"}

@app.post("/add_embeddings/")
async def add_embeddings(
    user_id: str = Form(...),
    pdf_list: List[UploadFile] = File(...),
    doc_ids: List[str] = Form(...),
):
    try:
        print(f"Received user_id: {user_id}")
        print(f"Received doc_ids: {doc_ids}")
        print(f"Number of files: {len(pdf_list)}")
        for file, doc_id in zip(pdf_list, doc_ids):
            add_docs(file, user_id, doc_id)  
        return {"status": "success", "message": "Embeddings added successfully."}
    except Exception as e:
        return {"status": "fail", "message": f"Failed to add embeddings. Error occurred: {e}"}

class QueryModel(BaseModel):
    query: str
    user_id: str
    doc_ids: List[str]

@app.post("/qa_rag/")
def qa_rag(query_model: QueryModel):
    query = query_model.query
    user_id = query_model.user_id
    doc_ids = query_model.doc_ids
    replies = []
    for id in doc_ids:
        reply = Haystack_qa_1(query, user_id, id)
        replies.append(reply)
    return {"status":"success", 'result' : replies}



#VECTOR_STORE.delete(delete_all=True, namespace="user_1")
#add_docs("OWASP Application Security Verification Standard 4.0.3-en.pdf", "pdfs", 'user_1', 1)
#add_docs("file_87.pdf", "pdfs", 'user_1', 2)
#Haystack_qa_1("a", "b", "What is authorization?", 'user_1', 1)

#submit_docs_for_rag("OWASP Application Security Verification Standard 4.0.3-en.pdf", "pdfs")
#MONGODB_COLLECTION.delete_many({})
