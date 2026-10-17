"""
rag_pipeline.py
================
Core Retrieval-Augmented Generation (RAG) logic for the Document Q&A app using Google Gemini (Google AI Studio).

This module is intentionally UI-agnostic so it can be unit-tested or reused
outside of Streamlit. It exposes:

* ``load_document``       -> parse an uploaded file (PDF / DOCX / TXT) into LangChain Documents
* ``split_documents``     -> chunk documents with ``RecursiveCharacterTextSplitter``
* ``RAGPipeline``         -> embeddings + FAISS vector store + retrieval + streaming generation with Google Gemini
* ``get_default_api_key`` -> helper to load Gemini API key from environment

The pipeline never touches Streamlit APIs, which keeps the concerns separated:
``app.py`` handles the UI, this file handles the "brains".
"""

from __future__ import annotations

import io
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Supported upload extensions (lower-case, including the dot).
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

#: Default embedding model used to vectorise chunks (Gemini standard embedding).
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-2"

#: Default chat model used to synthesise the final answer (Gemini 2.5 Flash).
DEFAULT_CHAT_MODEL = "gemini-2.5-flash"

#: Keywords to exclude specialized media/robotic endpoints from general text RAG.
EXCLUDED_CHAT_KEYWORDS = {"tts", "image", "audio", "robotics", "computer-use", "imagen", "embed"}

#: System prompt template. The retrieved context is injected into ``{context}``.
SYSTEM_PROMPT_TEMPLATE = """You are a helpful document question-answering assistant.

Answer the user's question using ONLY the context provided below. If the answer
cannot be found in the context, say honestly that the document does not contain
that information — do not make anything up. Be concise and, when useful, quote
the relevant part of the document.

Context:
{context}
"""


# ---------------------------------------------------------------------------
# Dynamic Model Discovery
# ---------------------------------------------------------------------------

def discover_gemini_models(api_key: str) -> Tuple[List[str], List[str]]:
    """Query Google AI Studio dynamically to discover active text models for this key."""
    chat_models: List[str] = []
    embed_models: List[str] = []
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        for m in client.models.list():
            raw_name = getattr(m, "name", "") or ""
            clean_name = raw_name.split("/")[-1].lower()
            
            # Skip specialized non-text models (TTS, image generation, robotics)
            if any(kw in clean_name for kw in EXCLUDED_CHAT_KEYWORDS):
                if "embed" in clean_name:
                    embed_models.append(clean_name)
                continue

            actions = [str(a).lower() for a in (getattr(m, "supported_actions", []) or [])]
            if any("generatecontent" in a for a in actions) or ("gemini" in clean_name and "embed" not in clean_name):
                chat_models.append(clean_name)
            if any("embedcontent" in a for a in actions) or "embed" in clean_name:
                embed_models.append(clean_name)
    except Exception:
        pass

    # Standard high-throughput general text models (in prioritized order)
    priority_chat = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-2.5-flash-lite",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ]
    priority_embed = [
        "gemini-embedding-001",
        "gemini-embedding-exp-03-07",
        "text-embedding-004",
    ]

    final_chat = list(dict.fromkeys(priority_chat + chat_models))
    final_embed = list(dict.fromkeys(priority_embed + embed_models))
    return final_chat, final_embed


# ---------------------------------------------------------------------------
# Document loading / parsing
# ---------------------------------------------------------------------------

class DocumentLoadError(Exception):
    """Raised when an uploaded document cannot be parsed."""


def _load_pdf(data: bytes, source_name: str) -> List[Document]:
    """Parse a PDF byte stream into one ``Document`` per page using ``pypdf``."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise DocumentLoadError("pypdf is not installed. Run: pip install pypdf") from exc

    documents: List[Document] = []
    try:
        reader = PdfReader(io.BytesIO(data))
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": source_name, "page": page_number},
                    )
                )
    except Exception as exc:  # corrupted / encrypted PDFs
        raise DocumentLoadError(f"Failed to read PDF '{source_name}': {exc}") from exc

    if not documents:
        raise DocumentLoadError(
            f"No extractable text found in '{source_name}'. "
            "It may be a scanned/image-only PDF."
        )
    return documents


def _load_docx(data: bytes, source_name: str) -> List[Document]:
    """Parse a DOCX byte stream into a single ``Document`` using ``python-docx``."""
    try:
        import docx  # python-docx
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise DocumentLoadError(
            "python-docx is not installed. Run: pip install python-docx"
        ) from exc

    try:
        document = docx.Document(io.BytesIO(data))
        paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
    except Exception as exc:  # corrupted / not a real docx
        raise DocumentLoadError(f"Failed to read DOCX '{source_name}': {exc}") from exc

    text = "\n".join(paragraphs)
    if not text.strip():
        raise DocumentLoadError(f"No extractable text found in '{source_name}'.")

    return [Document(page_content=text, metadata={"source": source_name, "page": 1})]


def _load_txt(data: bytes, source_name: str) -> List[Document]:
    """Parse a plain-text byte stream into a single ``Document``."""
    # utf-8 first, then a lenient fallback for exotic encodings.
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:  # pragma: no cover - latin-1 never raises, kept for safety
        raise DocumentLoadError(f"Could not decode text file '{source_name}'.")

    if not text.strip():
        raise DocumentLoadError(f"'{source_name}' is empty.")

    return [Document(page_content=text, metadata={"source": source_name, "page": 1})]


def load_document(uploaded_file) -> List[Document]:
    """Route an uploaded file to the correct parser based on its extension.

    Args:
        uploaded_file: A Streamlit ``UploadedFile`` (or any object exposing
            ``.name`` and ``.read()`` returning bytes).

    Returns:
        A list of LangChain ``Document`` objects with ``source``/``page`` metadata.

    Raises:
        DocumentLoadError: If the extension is unsupported or parsing fails.
    """
    filename = getattr(uploaded_file, "name", "") or ""
    extension = os.path.splitext(filename)[1].lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise DocumentLoadError(
            f"Unsupported file format '{extension or 'unknown'}'. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
        )

    try:
        data = uploaded_file.read()
    except Exception as exc:
        raise DocumentLoadError(f"Could not read the uploaded file: {exc}") from exc

    if not data:
        raise DocumentLoadError(f"'{filename}' appears to be empty.")

    parsers = {".pdf": _load_pdf, ".docx": _load_docx, ".txt": _load_txt}
    return parsers[extension](data, filename)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def split_documents(
    documents: Sequence[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Document]:
    """Split documents into overlapping chunks suitable for embedding.

    Overlap preserves context that would otherwise be cut at chunk boundaries,
    which improves retrieval quality.

    Args:
        documents: Source documents to split.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Characters shared between consecutive chunks.

    Returns:
        The list of chunked ``Document`` objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(list(documents))
    # Drop chunks that are only whitespace to avoid polluting the index.
    return [c for c in chunks if c.page_content.strip()]


# ---------------------------------------------------------------------------
# RAG pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """Embedding + FAISS vector store + retrieval + streaming Google Gemini LLM generation.

    Typical usage::

        pipeline = RAGPipeline(api_key="AIzaSy...")
        pipeline.build_index(load_document(uploaded_file))
        docs = pipeline.retrieve("What is the objective?")
        for token in pipeline.stream_answer("What is the objective?"):
            print(token, end="")
    """

    def __init__(
        self,
        api_key: str,
        chat_model: str = DEFAULT_CHAT_MODEL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        top_k: int = 4,
        temperature: float = 0.0,
    ) -> None:
        """Initialise Google Gemini embeddings, chat model and (lazily) vector store.

        Args:
            api_key: Google Gemini API key.
            chat_model: Chat completion model name (e.g., "gemini-2.5-flash").
            embedding_model: Embedding model name (e.g., "gemini-embedding-001").
            chunk_size: Characters per chunk used when indexing.
            chunk_overlap: Overlap between chunks used when indexing.
            top_k: Number of chunks to retrieve per query.
            temperature: Sampling temperature for the chat model.
        """
        if not api_key or not api_key.strip():
            raise ValueError("A valid Google Gemini API key is required.")

        self.api_key = api_key.strip()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.temperature = temperature

        clean_embedding_model = embedding_model.removeprefix("models/") if embedding_model else DEFAULT_EMBEDDING_MODEL
        clean_chat_model = chat_model.removeprefix("models/") if chat_model else DEFAULT_CHAT_MODEL
        self.chat_model_name = clean_chat_model
        self.embedding_model_name = clean_embedding_model

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=clean_embedding_model,
            google_api_key=self.api_key,
        )
        self.llm = ChatGoogleGenerativeAI(
            model=clean_chat_model,
            google_api_key=self.api_key,
            temperature=temperature,
        )
        self.vector_store: Optional[FAISS] = None

    # -- indexing ----------------------------------------------------------

    def build_index(self, documents: Sequence[Document]) -> int:
        """Chunk, embed and index documents into an in-memory FAISS store.

        Supports dynamic discovery and fallback across standard Gemini embedding models.

        Args:
            documents: Parsed source documents.

        Returns:
            The number of chunks that were indexed.

        Raises:
            ValueError: If no usable text was produced.
        """
        if not documents:
            raise ValueError("No documents were provided for indexing.")

        chunks = split_documents(documents, self.chunk_size, self.chunk_overlap)
        if not chunks:
            raise ValueError("The document produced no text chunks to index.")

        _, dynamic_embed_models = discover_gemini_models(self.api_key)
        candidate_models = [self.embedding_model_name] + dynamic_embed_models
        unique_candidates = list(dict.fromkeys(candidate_models))

        last_error = None
        for model_name in unique_candidates:
            try:
                emb = GoogleGenerativeAIEmbeddings(
                    model=model_name,
                    google_api_key=self.api_key,
                )
                self.vector_store = FAISS.from_documents(chunks, emb)
                self.embeddings = emb
                self.embedding_model_name = model_name
                return len(chunks)
            except Exception as exc:
                last_error = exc
                if any(err in str(exc).lower() for err in ("404", "not_found", "not found", "429", "resource_exhausted")):
                    continue
                raise exc

        if last_error is not None:
            raise last_error
        return len(chunks)

    # -- retrieval ---------------------------------------------------------

    def retrieve(self, question: str) -> List[Document]:
        """Return the ``top_k`` most similar chunks for ``question``.

        Args:
            question: The user's natural-language query.

        Returns:
            A list of relevant ``Document`` chunks (empty if not indexed).
        """
        if self.vector_store is None:
            raise RuntimeError("The vector store has not been built yet.")
        return self.vector_store.similarity_search(question, k=self.top_k)

    @staticmethod
    def format_sources(documents: Sequence[Document]) -> List[Dict[str, object]]:
        """Convert retrieved ``Document`` objects into lightweight source dicts.

        This makes the sources easy to render in the UI and cheap to store in
        ``st.session_state`` (we keep a short snippet rather than full text).
        """
        sources: List[Dict[str, object]] = []
        for index, doc in enumerate(documents, start=1):
            snippet = doc.page_content.strip()
            if len(snippet) > 400:
                snippet = snippet[:400] + "..."
            sources.append(
                {
                    "index": index,
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page", "-"),
                    "content": snippet,
                }
            )
        return sources

    # -- generation --------------------------------------------------------

    @staticmethod
    def _format_context(documents: Sequence[Document]) -> str:
        """Concatenate retrieved chunks into a single context block for the LLM."""
        blocks = []
        for i, doc in enumerate(documents, start=1):
            location = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page")
            if page is not None:
                location = f"{location} (page {page})"
            blocks.append(f"[{i}] {location}\n{doc.page_content.strip()}")
        return "\n\n".join(blocks) if blocks else "No relevant context was found."

    @staticmethod
    def _build_messages(
        context: str,
        question: str,
        chat_history: Optional[Iterable[Dict[str, str]]] = None,
    ) -> List:
        """Assemble the message list sent to the chat model.

        Args:
            context: Retrieved document context.
            question: The current user question.
            chat_history: Prior turns as ``{"role": ..., "content": ...}`` dicts.

        Returns:
            A list of LangChain chat message objects.
        """
        messages: List = [
            SystemMessage(content=SYSTEM_PROMPT_TEMPLATE.format(context=context))
        ]
        for turn in chat_history or []:
            role = turn.get("role")
            content = turn.get("content", "")
            if not content:
                continue
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        messages.append(HumanMessage(content=question))
        return messages

    def stream_answer(
        self,
        question: str,
        chat_history: Optional[Iterable[Dict[str, str]]] = None,
        context: Optional[str] = None,
    ) -> Iterable[str]:
        """Yield the assistant's answer token-by-token (for ``st.write_stream``).

        Dynamically queries available models and streams tokens.

        Args:
            question: The user's question.
            chat_history: Prior conversation turns for follow-up awareness.
            context: Pre-formatted context; retrieved automatically if omitted.

        Yields:
            Successive text fragments of the answer.
        """
        if context is None:
            context = self._format_context(self.retrieve(question))

        messages = self._build_messages(context, question, chat_history)

        dynamic_chat_models, _ = discover_gemini_models(self.api_key)
        candidate_chat_models = [self.chat_model_name] + dynamic_chat_models
        unique_candidates = list(dict.fromkeys(candidate_chat_models))

        last_error = None
        for model_name in unique_candidates:
            try:
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=self.api_key,
                    temperature=self.temperature,
                )

                stream_iter = llm.stream(messages)

                def extract_tokens():
                    for chunk in stream_iter:
                        content = getattr(chunk, "content", "")
                        if isinstance(content, str) and content:
                            yield content
                        elif isinstance(content, list):
                            for item in content:
                                if isinstance(item, str) and item:
                                    yield item
                                elif isinstance(item, dict) and "text" in item:
                                    yield item["text"]

                token_gen = extract_tokens()
                first_token = next(token_gen, None)
                if first_token is not None:
                    self.llm = llm
                    self.chat_model_name = model_name
                    yield first_token
                    yield from token_gen
                    return
            except Exception as exc:
                last_error = exc
                err_str = str(exc).lower()
                if any(err in err_str for err in ("404", "not_found", "not found", "429", "resource_exhausted", "quota")):
                    continue
                yield f"\n\n**Error while generating the answer:** {exc}"
                return

        yield f"\n\n**Error while generating the answer:** {last_error or 'No available Gemini chat models found for your API key.'}"


def get_default_api_key() -> str:
    """Read the Gemini API key from environment variables (``GEMINI_API_KEY`` or ``GOOGLE_API_KEY``)."""
    return (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
