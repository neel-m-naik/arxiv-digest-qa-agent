import os
import re
import time
import tempfile
import urllib.request
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import arxiv
import pymupdf  # Modern PyMuPDF import
import chromadb
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

GEMINI_MODEL = "gemini-3.5-flash-lite"

# Reuse a single GenAI client instance across invocations
_genai_client: Optional[genai.Client] = None


def _get_genai_client() -> genai.Client:
    global _genai_client
    if _genai_client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in environment or .env file.")
        _genai_client = genai.Client(api_key=api_key)
    return _genai_client


@dataclass
class AgentState:
    raw_query: str                          # User's initial input
    query_type: str = ""                    # 'arxiv_id' or 'topic'
    arxiv_id: Optional[str] = None          # Resolved arXiv ID
    paper_metadata: Dict[str, Any] = field(default_factory=dict)
    pdf_path: Optional[str] = None          # Local path to downloaded PDF
    extracted_text: str = ""                # Extracted text from PDF
    briefing: str = ""                      # Formatted Executive Briefing
    vector_collection: Any = None           # In-memory Chroma collection
    chat_history: List[Dict[str, str]] = field(default_factory=list)
    error: Optional[str] = None             # Error tracking


# --- Helper: Resilient LLM Invocation ---
def call_llm_with_retry(prompt: str) -> str:
    client = _get_genai_client()
    config = types.GenerateContentConfig(temperature=0.2)

    last_error: Optional[Exception] = None
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=config
            )
            if response and response.text:
                return response.text.strip()
            last_error = RuntimeError("Model returned an empty response.")
        except Exception as e:
            last_error = e

        if attempt < 2:
            time.sleep(2 * (attempt + 1))

    raise RuntimeError(f"LLM request failed after retries: {last_error}")


# --- Node 1: Query Understanding ---
def node_query_understanding(state: AgentState) -> AgentState:
    query = state.raw_query.strip()
    arxiv_pattern = r"(\d{4}\.\d{4,5}(v\d+)?)"
    match = re.search(arxiv_pattern, query)
    
    if match:
        state.query_type = "arxiv_id"
        state.arxiv_id = match.group(1)
    else:
        state.query_type = "topic"
    return state


# --- Node 2 & 3: arXiv Retrieval & Selection ---
def node_arxiv_retrieval(state: AgentState) -> AgentState:
    client = arxiv.Client(num_retries=3)

    try:
        if state.query_type == "arxiv_id":
            search = arxiv.Search(id_list=[state.arxiv_id])
        else:
            search = arxiv.Search(
                query=state.raw_query,
                max_results=3,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
        results = list(client.results(search))
        if not results:
            state.error = f"No papers found matching: {state.raw_query}"
            return state
            
        paper = results[0]
        raw_id = paper.entry_id.split("/")[-1] if hasattr(paper, "entry_id") else state.arxiv_id or "downloaded_paper"
        
        state.paper_metadata = {
            "title": paper.title.replace("\n", " ").strip(),
            "authors": [a.name for a in paper.authors],
            "published": paper.published.strftime("%Y-%m-%d"),
            "arxiv_id": raw_id,
            "pdf_url": paper.pdf_url,
            "summary": paper.summary.replace("\n", " ").strip()
        }

        download_dir = tempfile.mkdtemp(prefix="arxiv_agent_")
        target_path = os.path.join(download_dir, "current_paper.pdf")

        # Handle both client.download_pdf and direct urllib download safely
        try:
            client.download_pdf(paper, dirpath=download_dir, filename="current_paper.pdf")
            state.pdf_path = target_path
        except (AttributeError, Exception):
            req = urllib.request.Request(
                paper.pdf_url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            with urllib.request.urlopen(req) as response, open(target_path, "wb") as f:
                f.write(response.read())
            state.pdf_path = target_path

        if not os.path.exists(state.pdf_path) or os.path.getsize(state.pdf_path) == 0:
            state.error = "Downloaded PDF file is empty or missing."

    except Exception as e:
        state.error = f"arXiv retrieval error: {str(e)}"
        
    return state


# --- Node 4: PDF Parsing ---
def node_parse_pdf(state: AgentState) -> AgentState:
    if state.error or not state.pdf_path:
        return state

    doc = None
    try:
        doc = pymupdf.open(state.pdf_path)
        pages_text = [page.get_text() for page in doc]
        state.extracted_text = "\n".join(pages_text)
        
        print(f"    [debug] Extracted {len(state.extracted_text)} characters from {len(pages_text)} pages.")
        
        if len(state.extracted_text.strip()) < 150:
            state.error = "PDF extracted insufficient text (may be scanned or corrupted layout)."
    except Exception as e:
        state.error = f"PDF parsing failed: {str(e)}"
    finally:
        if doc is not None:
            doc.close()
        
    return state


# --- Node 5: Chunking & Vector DB (Chroma) ---
def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    if not text:
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def node_chunk_and_embed(state: AgentState) -> AgentState:
    if state.error or not state.extracted_text:
        return state
        
    try:
        # Prepend title and abstract for comprehensive retrieval
        corpus = (
            f"Paper Title: {state.paper_metadata.get('title', '')}\n"
            f"Abstract: {state.paper_metadata.get('summary', '')}\n\n"
            f"{state.extracted_text}"
        )
        
        chunks = chunk_text(corpus)
        print(f"    [debug] Indexed {len(chunks)} text chunks into Chroma vector store.")
        
        chroma_client = chromadb.Client()
        collection_name = "current_paper_collection"
        
        # Reset collection to avoid retaining stale vectors
        try:
            chroma_client.delete_collection(collection_name)
        except Exception:
            pass
            
        collection = chroma_client.create_collection(name=collection_name)
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        collection.add(documents=chunks, ids=ids)
        
        state.vector_collection = collection
    except Exception as e:
        state.error = f"Vector database error: {str(e)}"
        
    return state


# --- Node 6: Summarize (Executive Briefing) ---
def node_generate_briefing(state: AgentState) -> AgentState:
    if state.error:
        return state
        
    try:
        meta = state.paper_metadata
        context = f"Abstract: {meta.get('summary', '')}\n\n" + state.extracted_text[:6000]
        
        prompt = f"""
        You are an AI research briefing agent. Produce a structured executive briefing for the research paper below.
        Format your response in clean Markdown matching these sections:
        
        # Executive Briefing: {meta.get('title', 'Untitled')}
        - **Authors:** {', '.join(meta.get('authors', []))}
        - **arXiv ID:** {meta.get('arxiv_id', 'N/A')}
        - **Published:** {meta.get('published', 'N/A')}
        - **Link:** {meta.get('pdf_url', 'N/A')}
        
        ### 1-Paragraph Plain-English Summary
        (Provide a clear summary explaining why this paper matters)
        
        ### Problem Statement
        (What specific challenge or limitation are the authors addressing?)
        
        ### Method / Approach
        (Bulleted list explaining the core methodology or system architecture)
        
        ### Key Results & Claims
        (Primary metrics, performance gains, or conceptual findings)
        
        ### Limitations
        (Explicitly state any constraints, untested assumptions, or failure points noted)
        
        ### Suggested Follow-Up Questions
        (3-4 insightful questions a researcher might ask)
        
        Paper Content:
        {context}
        """
        
        state.briefing = call_llm_with_retry(prompt)
    except Exception as e:
        state.error = f"Briefing generation error: {str(e)}"
        
    return state


# --- Node 7: Grounded QA Loop (RAG) ---
def answer_question(state: AgentState, user_query: str) -> str:
    if state.vector_collection is None:
        return "No vector collection available in state."
    if not user_query.strip():
        return "Please enter a question."
        
    try:
        # Retrieve top 6 chunks for broad context coverage
        results = state.vector_collection.query(query_texts=[user_query], n_results=6)
        matched_chunks = results.get('documents', [[]])[0]
        
        base_context = (
            f"Paper Title: {state.paper_metadata.get('title', '')}\n"
            f"Abstract: {state.paper_metadata.get('summary', '')}\n\n"
            "Retrieved Passages from Paper:\n"
            + "\n---\n".join(matched_chunks)
        )
        
        prompt = f"""
You are an expert research assistant answering questions about an academic paper.
Use the provided Paper Context (Title, Abstract, and Retrieved Passages) to answer the user's question accurately.

Paper Context:
{base_context}

User Question:
{user_query}

Instructions:
- If the question can be answered using the provided context, give a concise, accurate response grounded directly in the text.
- If the question asks about something completely unrelated to or unmentioned in the paper (e.g. quantum computing, irrelevant fields), respond with: "I cannot find this information in the paper."
- Do not make up facts or hallucinate.
"""
        
        answer = call_llm_with_retry(prompt)
        state.chat_history.append({"question": user_query, "answer": answer})
        return answer
    except Exception as e:
        return f"Error answering question: {str(e)}"