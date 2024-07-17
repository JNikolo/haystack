import datetime
import time
from fastapi import FastAPI, HTTPException, UploadFile, Form, File, Cookie, Depends, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Annotated, List, Any, Dict, Tuple

import uvicorn

#useful libraries
from fastapi.responses import HTMLResponse
from icecream import ic
import pprint
from pypdf import PdfReader
import os
from dotenv import load_dotenv
from pydantic import BaseModel


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

#DBs firebase for frontend account auth and pinecone for backend pdf embeddings
from langchain_pinecone import PineconeVectorStore
import firebase_admin
from firebase_admin.auth import verify_id_token, create_session_cookie

load_dotenv()

#Configure gemini api
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

# Hugging Face model for embeddings.
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device': 'cpu'} #cuda if gpu is needed
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
    "localhost:5173",
    'https://localhost:5173',
    'http://127.0.0.1:5173',
    'https://127.0.0.1:5173'
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Authentication
GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
firebase_admin.initialize_app(credential=firebase_admin.credentials.Certificate(GOOGLE_APPLICATION_CREDENTIALS))

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
    reader = PdfReader(submitted_pdf.file)
    raw_text = ''
    for page in reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    docs = text_splitter.split_text(raw_text)#, metadatas = [{'doc_id' : 1}])

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
    print(f"Creating vector store for {len(pdf)} chunks")
    VECTOR_STORE.add_texts(
        texts = pdf,
        namespace = namespace,
        metadatas = [{'doc_id' : doc_id} for _ in range(len(pdf))]
    )


def Haystack_qa_1(query: str, namespace:str, doc_id:int):
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

########################################################################################################################
######################################################## ROUTES ########################################################
########################################################################################################################

#################################################Dependency###########################################
#@app.post('/verify')
async def verify_user(request: Request):
    try:
        print("Request Headers: ", request.cookies)
        token = request.cookies.get('session')
        print("Token received: ", token)
        if not token:
            raise HTTPException(status_code=401, detail="UNAUTHORIZED REQUEST!")
        
        decoded_claims = firebase_admin.auth.verify_session_cookie(token)
        print("Returing decoded claims")
        return decoded_claims
    
    except firebase_admin.auth.InvalidIdTokenError:
        print("unauthorized")
        raise HTTPException(status_code=401, detail="UNAUTHORIZED REQUEST!")

    except firebase_admin.auth.ExpiredIdTokenError:
        print("session expired")
        raise HTTPException(status_code=401, detail="EXPIRED SESSION")

    except Exception as e: 
        print("problem")
        print(e)
        raise HTTPException(status_code=500, detail="Internal Server Error")

#################################################Routes###################################################

# #route to the /querydocuments page
# @app.get("/querydocuments/{query}/topk/{top_k}")
# def read_item(query: str, top_k: int):
#     return {"q": query, "top_k": top_k}

# @app.post("/uploadfiles/")
# async def create_upload_files(files: list[UploadFile]):
#     try:
#         for file in files:
#             reader = PdfReader(file.file)
#             text = ""
#             for page_num in range(len(reader.pages)):
#                 text += reader.pages[page_num].extract_text()
#             # Save the extracted text to a .txt file
#             # For now it is saved in the "texts" directory
#             # Eventually, we will need to store it in a database or file system
#             output_filename = "texts/" + file.filename.replace(".pdf", ".txt")
#             with open(output_filename, "w", encoding="utf-8") as txt_file:
#                 txt_file.write(text)
#         return {"status": "success", "message": "Text extracted successfully."}
#     except Exception as e:
#         return {"status": "fail", "message": f"Failed to extract text. Error ocurred: {e}"}
    
@app.post("/searchkeyword/", dependencies=[Depends(verify_user)])
async def search_keyword(files: List[UploadFile], keyword: str) -> Dict[str, Any]:
    try:
        keyword_counts, total = await search_keyword_in_pdfs(files, keyword)
        return {"status":"success", "keyword": keyword, "total": total, "results": keyword_counts}
    except Exception as e:
        return {"status":"fail", "keyword": keyword, "message": f"Failed to search for keyword. Error ocurred: {e}"}

@app.post("/conceptsfrequencies/", dependencies=[Depends(verify_user)])
async def concept_frequencies(files: List[UploadFile]):

    try:
        results = await concepts_frequencies_in_pdfs(files)
        return {"status": "success", "message": "Concept frequencies extracted successfully.", "results": results}
    except Exception as e:
        return {"status": "fail", "message": f"Failed to extract concepts. Error ocurred: {e}"}


@app.post("/topicmodeling/", dependencies=[Depends(verify_user)])
async def topic_modeling(files: List[UploadFile]):
    try:
        df_count, df_importance = await topic_modeling_from_pdfs(files)
        count = df_count.to_dict(orient='records')
        importance = df_importance.to_dict(orient='records')
        result = {"count": count, "importance": importance}
        
        return {"status": "success", "message": "Topic modeling performed successfully.", "result": result}
    except Exception as e:
        return {"status": "fail", "message": f"Failed to perform topic modeling. Error ocurred: {e}"}

@app.delete("/delete_embeddings/")
def delete_embeddings(user_data: Annotated[dict, Depends(verify_user)],): 
    try:
        user_id = user_data['uid']
        VECTOR_STORE.delete(delete_all=True, namespace=user_id)
        return {"status": "success", "message": "Embeddings deleted successfully."}
    except Exception as e:
        return {"status": "fail", "message": f"Failed to delete embeddings. Error ocurred: {e}"}
   
@app.post("/add_embeddings/")
async def add_embeddings(
    #user_id: str = Form(...),
    user_data: Annotated[dict, Depends(verify_user)],
    pdf_list: List[UploadFile] = File(...),
    doc_ids: List[str] = Form(...),
    #data: Annotated[Tuple[List[UploadFile], List[str], dict], Depends(get_data)]
    #request: Request
):
    try:
        # Verify user
        #user_data = await verify_user(request)
        #print(f"User Data: {user_data}")

        # Get form data
        #form = await request.form()
        #doc_ids = form.getlist("doc_ids")
       # pdf_list = form.getlist("pdf_list")
        print(f"Received user_id: {user_data['uid']}")
        print(f"Received doc_ids: {doc_ids}")
        print(f"Number of files: {len(pdf_list)}")
        for file, doc_id in zip(pdf_list, doc_ids):
            add_docs(file, user_data['uid'], doc_id)  
        return {"status": "success", "message": "Embeddings added successfully."}
    except Exception as e:
        return {"status": "fail", "message": f"Failed to add embeddings. Error occurred: {e}"}

class QARAGRequest(BaseModel):
    query: str
    #user_id: str
    doc_ids: List[str]

@app.post("/qa_rag/")
def qa_rag(request: QARAGRequest, user_data: Annotated[dict, Depends(verify_user)],):
    replies = []
    user_id = user_data['uid']
    for id in request.doc_ids:
        reply = Haystack_qa_1(request.query, user_id, id)
        replies.append(reply)
    return {"status":"success", 'result' : replies}

#VECTOR_STORE.delete(delete_all=True, namespace='user_1')
#VECTOR_STORE.delete(delete_all=True, namespace='user_2')


########################################################################################################################
#Session cookies

class sessionModel(BaseModel):
    idToken: str
#    csrfToken: str

#create seesion cookie
def create_session_cookie(id_token, expires_in):
    # Implement your logic to create a session cookie based on the ID token
    # Example implementation:
    session_cookie = firebase_admin.auth.create_session_cookie(id_token, expires_in=expires_in)
    return session_cookie

@app.post("/session_login/")
async def session_login(session_model: sessionModel):#, csrf_token: str = Depends()):
    id_token = session_model.idToken
    #csrf = session_model.csrfToken
     # Verify CSRF token
    #if not csrf_token or csrf != csrf_token:
    #    raise HTTPException(status_code=401, detail="UNAUTHORIZED REQUEST!")

    expires_in = datetime.timedelta(days=5)

    # To ensure that cookies are set only on recently signed in users, check auth_time in
    # ID token before creating a cookie.
    print("id token obtained: ", id_token)
    try:
        print("entered try")
        decoded_claims = verify_id_token(id_token)
        print("decoded id")
        # Only process if the user signed in within the last 5 minutes.
        # if (time.time() - decoded_claims['auth_time']) < (5 * 60):
        expires = datetime.datetime.now(datetime.timezone.utc) + expires_in

        session_cookie = create_session_cookie(id_token, expires_in=expires_in)

        print("Session cookie: ", session_cookie)

        response = JSONResponse({'status': 'success'})

        #response.headers["Set-Cookie"] = "Secure; HttpOnly; Partitioned"
        expire_time = expires.strftime('%a, %d %b %Y %H:%M:%S GMT')
        response.headers["Set-Cookie"] = f"session={session_cookie}; Expires={expire_time}; Secure; HttpOnly; Partitioned; SameSite=None; Path=/;"
        #response.set_cookie(
        #    key='session', value=session_cookie, domain=".localhost",path="/",expires=expires, httponly=True, secure=False, samesite="None")
        print("exiting try")
        return response
        # User did not sign in recently. To guard against ID token theft, require
        # re-authentication.
        # else:
        #     # User did not sign in recently, require re-authentication
        #     print("in else")
        #     raise HTTPException(status_code=401, detail="Recent sign in required")
    except firebase_admin.auth.InvalidIdTokenError:
        print("Invalid ID Token")
        raise HTTPException(status_code=401, detail="UNAUTHORIZED REQUEST!")

    except Exception as e: 
        print(f"Error in session_login: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

########################################################################################################################
#Verify Session cookies
'''
@app.get("/profile")
def access_restricted_content(session: str = Cookie(None)):
    if not session:
        return RedirectResponse(url='/login')

    try:
        # Verify the session cookie and check if it's revoked
        decoded_claims = firebase_admin.auth.verify_session_cookie(session, check_revoked=True)
        return serve_content_for_user(decoded_claims)
    
    except firebase_admin.auth.InvalidSessionCookieError:
        return RedirectResponse(url='/login')
'''
########################################################################################################################
#Session Logout
@app.post('/session_logout')
def session_logout(request: Request):
    token = request.cookies.get('session')
    print("Token in session logout: ", token)
    response = JSONResponse({'status': 'success'})
    try:
        if token:
            print("if started")
            decoded_claims = firebase_admin.auth.verify_session_cookie(token)
            firebase_admin.auth.revoke_refresh_tokens(decoded_claims['sub'])
            print("Refresh tokens revoked")
            #response = RedirectResponse(url='/login')
        #response.set_cookie('session', '', max_age=0)
        response.delete_cookie('session', path='/', httponly=True)
        print("Cookie deleted. Exiting session_logout")
        return response
    except Exception as e:
        print(f"Error in session_logout: {e}")
        response.delete_cookie('session', path='/', httponly=True)
        #response.delete_cookie('session', path='/session_login', httponly=True, secure=True, samesite="None")
        raise HTTPException(status_code=500, detail="Internal Server Error")

#VECTOR_STORE.delete(delete_all=True, namespace="user_1")
#add_docs("OWASP Application Security Verification Standard 4.0.3-en.pdf", "pdfs", 'user_1', 1)
#add_docs("file_87.pdf", "pdfs", 'user_1', 2)
#Haystack_qa_1("a", "b", "What is authorization?", 'user_1', 1)

#submit_docs_for_rag("OWASP Application Security Verification Standard 4.0.3-en.pdf", "pdfs")
#MONGODB_COLLECTION.delete_many({})

# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8000,) #ssl_keyfile="cert-key.pem", ssl_certfile="fullchain.pem")