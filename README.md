# Haystack project

Gathering insights from business documents is very time consuming, and exahusting. That is why we introduce Haystack, to make the journey easier and faster.

## How to run

First make sure to install the requirements running the code:
```
pip freeze -r requirements.txt
```

After the dependencies are installed, run the command:
```
fastapi dev app.py
```

To access the interactive API docs:
```
http://127.0.0.1:8000/docs 
```


## Features
- Query Documents: Users can chat with their own documents to get insights using the powerful Gemini LLM
- Authentication: Users can register and log in through Google's Firebase, which enhance security and scalability
- Document Analysis: Set of machine learning tools, to get more abstract or numerical insights from the documents.
  - Keyword counting: Tool that counts all the occurrences of a keyword (and its variants), and records the page location and adjacent text.
  - Topic modeling: Tool that helps visualizing the most imporant topics in the documents and their related keywords
  - Concept frequency: Tool that helps to visualize the most frequent concepts in the documents.

## Technologies
- Back-end: FastAPI, MongoDB
- Front-end: Flutter
- RAG: Langchain, Gemini LLM and embedding, MongoDB
- Authentication: Firebase
- Hosting: ?