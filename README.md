# 📄 Document Q&A (RAG with Google Gemini)

A Streamlit application that answers questions about your documents using
**Retrieval-Augmented Generation (RAG)** powered by **Google Gemini** and **LangChain**. Upload PDF, DOCX, or TXT files, and ask questions in a real-time chat interface — answers are grounded strictly in your document content with transparent source citations.

```
User → Streamlit UI → RAG Pipeline (FAISS Retrieval) → ChatGoogleGenerativeAI (Gemini 1.5 Flash) → Answer
```

---

## 🌟 Features

- 📎 **Multi-format Support:** Upload one or multiple **PDF, DOCX, or TXT** files.
- ⚡ **Google Gemini 2.5 Flash:** Fast, highly capable reasoning and streaming responses.
- 🔍 **Google gemini-embedding-001:** High quality embeddings for dense semantic retrieval.
- 🔒 **Secure Credential Handling:** Zero hardcoded keys, masked password inputs, and 3-tier prioritized API key resolution.
- 📚 **Transparent Source Attribution:** Expandable source viewer showing cited document chunks and page numbers.
- ♻️ **Session Caching:** Vector index stored in session state to prevent redundant re-indexing and token waste.

---

## 🛠️ Project Structure

| File | Purpose |
| :--- | :--- |
| `app.py` | Streamlit user interface, session state management, and secure API key resolution |
| `rag_pipeline.py` | Document loading, chunking, Gemini embeddings, FAISS vector store, and streaming LLM generation |
| `requirements.txt` | Python dependencies including `langchain-google-genai` and `google-generativeai` |
| `.env.example` | Template for configuring `GEMINI_API_KEY` |
| `.gitignore` | Prevents leaking credentials (`.env`, `secrets.toml`), caches, and local vector indices |

---

## 🔑 Obtaining a Free Gemini API Key

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Sign in with your Google account.
3. Click **Create API Key** (or **Get API key**).
4. Copy your newly generated API key (`AIzaSy...`).

---

## 🚀 Setup & Installation

### 1. Create and Activate a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Key (Optional / Recommended)

Copy `.env.example` to create your local `.env` file:

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Open `.env` and set your key:
```env
GEMINI_API_KEY=AIzaSy...your-actual-api-key
```

*(Alternatively, you can skip this step and paste your key directly into the sidebar in the running app.)*

### 4. Run the Application

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 💡 Usage Guide

1. **Provide API Key:** Enter your Gemini API key in the sidebar (if not loaded from `.env` or Streamlit secrets).
2. **Upload Documents:** Browse and select one or more PDF, DOCX, or TXT files.
3. **Process Documents:** Click **🚀 Process Document** to extract text, chunk content, generate Gemini embeddings, and build the FAISS index.
4. **Chat & Inquire:** Type a question or click any of the suggested sample questions.
5. **Inspect Sources:** Expand the **📚 Sources** dropdown beneath any answer to verify the retrieved passages and page numbers.

---

## 🔐 Security & Priority Key Resolution

The application resolves the Gemini API key in the following priority order:
1. **Sidebar Input** (Masked password field `type="password"`)
2. **Streamlit Secrets** (`.streamlit/secrets.toml` with `GEMINI_API_KEY` or `GOOGLE_API_KEY`)
3. **Environment Variables** (`.env` or system environment with `GEMINI_API_KEY` or `GOOGLE_API_KEY`)
