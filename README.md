# 🤖 Agentic AI RAG Assistant

An Agentic AI-powered Retrieval-Augmented Generation (RAG) chatbot that answers questions using information retrieved from a provided reference document.

The application combines **LangGraph**, **Pinecone**, **Sentence Transformers**, and **Groq LLM** to provide grounded and context-aware responses.

## 🚀 Live Demo

**Streamlit App:**  
https://agentic-aiassistant.streamlit.app/

## 📂 GitHub Repository

**Repository:**  
https://github.com/Bejjenki-Rakesh2003/agentic-ai-chatbot

---

## 📌 Project Overview

This project implements a RAG-based AI assistant that retrieves relevant information from a vector database and uses an LLM to generate answers based on the retrieved context.

The system follows an agentic workflow using **LangGraph**, where the question passes through different processing stages before the final response is generated.

The chatbot is designed to:

- Retrieve relevant information from the reference material
- Generate answers using retrieved context
- Reduce unsupported or hallucinated responses
- Display the retrieved sources used for generating an answer
- Provide a simple and interactive Streamlit interface

---

## 🧠 Architecture

The application follows this workflow:

    User Question
          │
          ▼
    Question Processing
          │
          ▼
    Pinecone Vector Search
          │
          ▼
    Relevant Context Retrieval
          │
          ▼
    LangGraph Workflow
          │
          ▼
    Groq LLM
          │
          ▼
    Grounded Answer
          │
          ▼
    Streamlit UI

---

## 🔄 RAG Workflow

### 1. Question

The user enters a question through the Streamlit interface.

### 2. Retrieval

The application searches the Pinecone vector database to retrieve the most relevant chunks from the reference material.

### 3. Generation

The retrieved context is passed to the Groq-powered LLM through the LangGraph workflow.

### 4. Grounding

The LLM generates an answer based on the retrieved reference material rather than relying only on general knowledge.

### 5. Source Display

The application provides an option to view the retrieved sources used during the response generation.

---

## 🛠️ Technologies Used

### Frontend / UI

- Streamlit

### AI / LLM

- Groq LLM
- `openai/gpt-oss-20b`

### Agentic Workflow

- LangGraph
- LangChain

### RAG / Vector Database

- Pinecone
- Retrieval-Augmented Generation

### Embeddings

- Sentence Transformers

### Document Processing

- PyPDF

### Configuration

- Python-dotenv
- Environment variables

---

## 📦 Main Dependencies

    streamlit
    langchain
    langchain-community
    langchain-text-splitters
    langgraph
    langchain-groq
    langchain-pinecone
    pinecone
    sentence-transformers
    pypdf
    python-dotenv
    pydantic

---

## 📁 Project Structure

    agentic_ai_rag/
    │
    ├── app.py
    ├── requirements.txt
    ├── README.md
    ├── .gitignore
    │
    └── src/
        ├── graph.py
        ├── retriever.py
        ├── llm.py
        └── ...

> The exact files inside `src/` may vary depending on the implementation.

---

## 🔐 Environment Variables

Create a `.env` file locally and add the required API credentials.

    GROQ_API_KEY=your_groq_api_key
    PINECONE_API_KEY=your_pinecone_api_key
    PINECONE_INDEX_NAME=your_pinecone_index_name

**Never commit your `.env` file or API keys to GitHub.**

The `.gitignore` file should include:

    .env
    venv/
    __pycache__/
    *.pyc

---

## 💻 Local Installation

### 1. Clone the repository

    git clone https://github.com/Bejjenki-Rakesh2003/agentic-ai-chatbot

### 2. Navigate to the project

    cd agentic_ai_rag

### 3. Create a virtual environment

    python -m venv venv

### 4. Activate the virtual environment

#### Windows

    venv\Scripts\activate

### 5. Install dependencies

    pip install -r requirements.txt

### 6. Configure environment variables

Create a `.env` file:

    GROQ_API_KEY=your_groq_api_key
    PINECONE_API_KEY=your_pinecone_api_key
    PINECONE_INDEX_NAME=your_pinecone_index_name

### 7. Run the application

    streamlit run app.py

The application will open at:

    http://localhost:8501

---

## 🌐 Deployment

The application is deployed using **Streamlit Community Cloud**.

### Live Application

https://agentic-aiassistant.streamlit.app/

The deployment uses the GitHub repository as the source and installs the dependencies specified in `requirements.txt`.

Required API keys should be configured through the deployment platform's secrets/environment-variable settings rather than being stored in the repository.

---

## 💬 Example Questions

The application can answer questions related to the provided Agentic AI reference material.

Example questions:

    What is Agentic AI?

    What are the building blocks of Agentic AI?

    What is multi-agent collaboration?

    How does Agentic AI differ from traditional AI?

---

## ✨ Key Features

- 🤖 Agentic AI workflow using LangGraph
- 🔎 Semantic retrieval using Pinecone
- 🧠 Sentence Transformer embeddings
- ⚡ Groq-powered LLM generation
- 📚 Reference-grounded responses
- 🔗 Retrieved source visibility
- 💬 Interactive Streamlit chatbot interface
- ☁️ Streamlit Cloud deployment
- 🔐 Environment-based API key management

---

## 🎯 Objective

The main objective of this project is to demonstrate how an **Agentic RAG system** can combine retrieval, reasoning, and LLM-based generation to answer questions using a specific knowledge source.

The project demonstrates practical usage of:

- Retrieval-Augmented Generation
- Vector databases
- Embedding models
- Agentic workflows
- LLM integration
- Source-grounded question answering

---

## 👩‍💻 Author

**Bejjenki Rakesh**

GitHub:  
https://github.com/Bejjenki-Rakesh2003

---

## 📄 Assignment Deliverables

### GitHub Repository

https://github.com/Bejjenki-Rakesh2003/agentic-ai-chatbot

### Published Application

https://agentic-aiassistant.streamlit.app/

---

## 🙏 Acknowledgement

This project was developed as part of an **AI Engineer / Agentic AI RAG assignment** to demonstrate practical implementation of LangGraph-based RAG systems with Pinecone and an LLM.
