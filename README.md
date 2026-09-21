# Autonomous arXiv Paper Digest & QA Agent

An agentic pipeline built in Python to autonomously ingest, parse, summarize, and provide grounded interactive question-answering over academic research papers from arXiv.

1. System Architecture & State Graph
The agent is structured as an explicit stateful graph where each node updates a shared AgentState object.

                  ┌───────────────────────┐
                  │ User Input: Query/ID  │
                  └───────────┬───────────┘
                              ▼
               ┌─────────────────────────────┐
               │ 1. Query Understanding Node │
               │ (Detect Topic vs. arXiv ID) │
               └──────────────┬──────────────┘
                              ▼
               ┌─────────────────────────────┐
               │ 2. arXiv Retrieval Node     │
               │ (API Search, Metadata, PDF) │
               └──────────────┬──────────────┘
                              ▼
               ┌─────────────────────────────┐
               │ 3. PDF Parsing Node         │
               │ (PyMuPDF Text Extraction)   │
               └──────────────┬──────────────┘
                              ▼
               ┌─────────────────────────────┐
               │ 4. Vector Store Setup Node  │
               │ (Chunking & Chroma DB)      │
               └──────────────┬──────────────┘
                              ▼
               ┌─────────────────────────────┐
               │ 5. Executive Briefing Node  │
               │ (Gemini LLM Synthesis)      │
               └──────────────┬──────────────┘
                              ▼
               ┌─────────────────────────────┐
               │ 6. Grounded QA Loop (RAG)   │
               │ (Chroma Search + Grounding) │
               └─────────────────────────────┘
Shared State Shape (AgentState)
raw_query: The original user string (topic or arXiv identifier).

query_type: Classified intent (arxiv_id or topic).

arxiv_id: Cleaned arXiv ID.

paper_metadata: Dictionary holding title, authors, publication date, URL, and summary.

pdf_path: Deterministic local path for downloaded PDF.

extracted_text: Raw extracted text across all PDF pages.

briefing: Structured Markdown executive summary.

vector_collection: In-memory Chroma collection instance containing indexed paper chunks.

chat_history: Running conversation turns during QA mode.

error: Tracking string for failure points.

2. Setup & Installation
Prerequisites
Python 3.10+

Free Gemini API Key (via Google AI Studio)

Installation
Clone this repository:

Bash
git clone 
cd arxiv_agent
Create and activate a virtual environment:

Bash
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
Install dependencies:

Bash
pip install arxiv pymupdf chromadb google-genai python-dotenv
Configure your environment variable:
Create a .env file in the project root:

Code snippet
GEMINI_API_KEY=your_gemini_api_key_here
Run the agent:

Bash
python main.py
3. Example Run
Input
Plaintext
2401.12345
Executive Briefing Output
Markdown
# Executive Briefing: Distributionally Robust Receive Combining
- **Authors:** J. Doe, et al.
- **arXiv ID:** 2401.12345
- **Published:** 2024-01-22
- **Link:** [https://arxiv.org/pdf/2401.12345](https://arxiv.org/pdf/2401.12345)

### 1-Paragraph Plain-English Summary
This paper introduces a distributionally robust receive combining framework for wireless communications that remains resilient against uncertainties in transmit signals, channel matrices, and impulse noises.

### Problem Statement
Traditional linear beamformers degrade heavily under pilot contamination, non-Gaussian channel impulse noises, and small sample sizes.

### Method / Approach
- Formulates a distributionally robust optimization objective over uncertainty sets.
- Employs linear estimators (diagonal loading, eigenvalue thresholding).
- Derives non-linear estimators operating in reproducing kernel Hilbert spaces (RKHS).

### Key Results & Claims
- Kernel-DL achieves significant MSE reduction (down to 0.53) under non-Gaussian impulse noise compared to classical Capon/ZF beamformers.
- Highlights practical runtime tradeoffs: Wiener-DL delivers real-time sub-millisecond execution compared to multi-second solvers.

### Limitations
- Computational overhead of exact distributionally robust solvers makes them prohibitive for high-mobility scenarios without approximations.
Sample Grounded QA Exchanges
Plaintext
Q: What is the primary method proposed by the authors?
A: Based on the provided paper context, the primary method proposed by the authors is a distributionally robust receive combining framework that is insensitive to various uncertainties (such as uncertainties in the transmit signal covariance, channel matrix, and channel noise covariance, as well as the existence of channel impulse noises, non-ideality of power amplifiers, and limited sample size of pilots)...

Q: What are the key numerical results or benchmark improvements reported?
A: 1. Impact of Pilot Size: Experimental results across Tables III–VI show that MSE decreases as pilot size increases.
   2. Effectiveness of Diagonal Loading: Significantly improves estimation performance under small pilot data sizes.
   3. Nonlinear Estimators: Kernel-DL achieves MSE of 0.66 (size 20) and 0.53 (size 50), outperforming linear beamformers under non-Gaussian noise.
   4. Computational Tradeoff: Wiener-DR exhibits runtimes between 4.22s to 7.19s, whereas Wiener-DL runs in sub-milliseconds.

Q: Does this paper discuss quantum computing hardware architectures?
A: I cannot find this information in the paper.
4. Design Decisions & Tradeoffs
1. In-Memory Ephemeral Vector DB (Chroma)
Decision: Chroma was selected to run in embedded client mode directly within memory.

Tradeoff: It requires no external server setup, making the tool portable and zero-configuration. The tradeoff is that vectors are not retained between separate CLI executions. However, because embedding a 10–20 page PDF takes only 2–3 seconds locally, re-indexing on each run preserves clean boundaries between papers.

2. PyMuPDF vs. Heavy OCR / Layout Engines
Decision: Used PyMuPDF (pymupdf) for direct stream extraction.

Tradeoff: PyMuPDF is fast (sub-second extraction across 15+ pages) with zero native compilation issues on Windows. The limitation is that it does not perform OCR on scanned image-only PDFs. A graceful failure check (len(extracted_text) < 150) detects this scenario.

3. State Object Architecture over Complex Orchestration Frameworks
Decision: Used an explicit AgentState dataclass passed sequentially through modular node functions.

Tradeoff: Rather than importing heavy workflow abstractions that hide state mutations, explicit state handling ensures transparency, deterministic node transitions, and simpler debugging.

4. Grounding & Anti-Hallucination Guardrails
Decision: Grounded QA combines Chroma passage retrieval with top-level metadata (title + abstract) and strict system-level instructions instructing the model to reply "I cannot find this information in the paper" when evidence is absent.

Tradeoff: This slightly increases prompt token count for each QA turn, but prevents false extrapolations on domain-specific academic literature.

What Would Be Done Differently With More Time
Implement section-aware semantic chunking (e.g., splitting strictly by \section{...} headers rather than token sliding windows).

Add support for multimodal diagram/table parsing using layout-aware models.