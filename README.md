
# 🤖 AI RAG Knowledge Assistant

A production-style **Retrieval-Augmented Generation (RAG)** application that answers questions using a custom knowledge base with full **source attribution, explainability, and confidence scoring**.

Built with **FastAPI, Streamlit, OpenAI, and FAISS**, this project demonstrates how modern AI systems combine retrieval and generation to produce reliable, grounded responses.

---

## 🚀 Demo

![App Screenshot](assets/demo.png)

---

## 🧠 Key Features

### 🔍 Retrieval-Augmented Generation (RAG)

- Retrieves relevant knowledge chunks using vector similarity search (FAISS)
- Grounds responses in real data instead of hallucinating

### 🧾 Source Attribution

- Displays exact documents and chunks used to generate the answer
- Highlights query-relevant terms for transparency

### 📊 Answer Confidence Scoring

- Dynamically scores answer reliability based on retrieved context
- Visual progress bar with interpretation (High / Medium / Low confidence)

### 🧠 Explainability Layer

- “Why this answer?” section explains how the system works
- Designed for trust and interpretability

### ⚡ FastAPI Backend

- REST API for query handling
- Health check endpoint for system monitoring

### 🎨 Streamlit Frontend

- Chat-style interface
- Example query buttons for quick demos
- Clean UX with structured sections

---

## 🏗️ Architecture

User Query
→ Streamlit UI
→ FastAPI Backend
→ OpenAI Embeddings (text-embedding-3-small)
→ FAISS Vector Store
→ Top-K Retrieved Chunks
→ LLM (gpt-4o-mini)
→ Grounded Answer + Sources

---

## 📂 Project Structure

app/api/ → FastAPI backend
app/ui/ → Streamlit frontend
app/rag/ → Retrieval + generation logic
app/data/ → Knowledge base files
app/embeddings/ → FAISS vector storage
assets/ → UI screenshots

---

## ⚙️ Tech Stack

- LLM: OpenAI (gpt-4o-mini)
- Embeddings: text-embedding-3-small
- Vector Store: FAISS
- Backend: FastAPI
- Frontend: Streamlit
- Language: Python

---

## 🔧 Setup Instructions

### 1. Clone the repository

git clone https://github.com/your-username/ai-rag-knowledge-assistant.git
cd ai-rag-knowledge-assistant

### 2. Create environment

conda create -n rag_env python=3.10
conda activate rag_env

### 3. Install dependencies

pip install -r requirements.txt

### 4. Set environment variable

export OPENAI_API_KEY=your_key_here

### 5. Run backend

uvicorn app.api.main:app --reload

### 6. Run frontend

streamlit run app/ui/streamlit_app.py

---

## 🧪 Example Queries

- What is customer churn?
- Why is churn important?
- What is customer retention?

---

## 📊 What This Project Demonstrates

- Building production-style AI systems
- Implementing RAG pipelines
- Designing explainable AI interfaces
- Integrating LLMs with vector databases
- Creating full-stack AI applications

---

## 🧠 Why This Matters

Traditional LLM apps hallucinate.

This system:

- retrieves real context
- grounds responses in data
- shows sources
- communicates confidence

This is how real-world AI systems are built.

---

## 🚀 Future Improvements

- Pinecone or Weaviate integration
- Authentication and user sessions
- Document upload pipeline
- Evaluation metrics
- Streaming responses

---

## 👤 Author

Darrell Mortalla
AI / ML Engineer

GitHub: https://github.com/dmortalla
Portfolio: (your site)
LinkedIn: (your link)

---

## ⭐ If you found this useful

Give it a star — it helps a lot!
