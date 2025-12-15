from __future__ import annotations

from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field

from core.state import Task, TaskResult, ConversationState
from tools.shared import toolify_agent, _cs

from llama_index.core import (
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.embeddings.openai import OpenAIEmbedding

def _get_index_dir() -> Path:
    here = Path(__file__).resolve().parent
    repo_root = here.parent 

    index_dir = (repo_root / "idc_index").resolve()

    print(f"[IDC RAG] Repo root:       {repo_root}")
    print(f"[IDC RAG] Index dir path:  {index_dir}")

    return index_dir


def _init_llamaindex_settings():
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")


def _load_existing_index():
    _init_llamaindex_settings()
    index_dir = _get_index_dir()

    if not index_dir.exists():
        raise FileNotFoundError(
            f"[IDC RAG] Index directory not found: {index_dir}. "
            "Run the external index builder script first."
        )

    docstore_path = index_dir / "docstore.json"
    if not docstore_path.exists():
        raise FileNotFoundError(
            f"[IDC RAG] docstore.json not found in {index_dir}. "
            "Your index build is incomplete or corrupted."
        )

    print("[IDC RAG] ✅ Loading existing index from disk...")
    storage_context = StorageContext.from_defaults(persist_dir=str(index_dir))
    index = load_index_from_storage(storage_context)
    return index


def _nodes_to_context(nodes, max_chars: int = 8000) -> str:
    chunks = []
    total = 0
    for n in nodes:
        text = n.get_content()
        if not text:
            continue
        remaining = max_chars - total
        if remaining <= 0:
            break
        snippet = text[:remaining]
        chunks.append(snippet)
        total += len(snippet)
    return "\n\n-----\n\n".join(chunks)


def _nodes_to_chunks(nodes, max_chars: int = 8000):
    chunks = []
    total = 0
    for n in nodes:
        text = n.get_content() or ""
        remaining = max_chars - total
        if remaining <= 0:
            break
        snippet = text[:remaining]
        total += len(snippet)
        chunks.append(
            {
                "text": snippet,
                "source": n.metadata.get("file_path") if hasattr(n, "metadata") else None,
                "score": getattr(n, "score", None),
            }
        )
    return chunks

class IDCWebQAAgent:
    name = "idc_web_qa"

    def __init__(self):
        index = _load_existing_index()
        self.index = index

    async def run(
        self,
        task: Task,
        state: ConversationState,
        top_k: int = 5,
        max_chars: int = 8000,
    ) -> TaskResult:
        question = task.user_msg

        retriever = self.index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(question)

        context = _nodes_to_context(nodes, max_chars=max_chars)
        chunk_list = _nodes_to_chunks(nodes, max_chars=max_chars)

        payload = {
            "question": question,
            "context": context,
            "chunks": chunk_list,
            "num_retrieved_chunks": len(nodes),
            "top_k": top_k,
            "max_chars": max_chars,
            "note": "Upstream agent should ground its answer in these chunks. No synthesis was done here.",
        }

        return TaskResult(output=payload, artifacts=payload)

_IDC_QA: Optional[IDCWebQAAgent] = None


def configure_idc_web_qa_tool():
    global _IDC_QA
    _IDC_QA = IDCWebQAAgent()


class IDCWebQAArgs(BaseModel):
    question: str = Field(..., description="Natural language question about IDC documentation.")
    top_k: int = Field(5, ge=1, le=20, description="How many chunks to retrieve from the IDC docs index.")
    max_chars: int = Field(8000, ge=500, le=20000, description="Max total characters to return across all retrieved chunks.")


@toolify_agent(
    name="idc_web_qa",
    description=(
        "Answers questions about the NCI Imaging Data Commons (IDC) using a local "
        "Markdown knowledge base that has been pre-indexed with LlamaIndex and "
        "stored in idc_index/."
    ),
    args_schema=IDCWebQAArgs,
    timeout_s=60,
)
async def idc_web_qa_runner(
    question: str,
    top_k: int = 5,
    max_chars: int = 8000,
):
    if _IDC_QA is None:
        raise RuntimeError(
            "IDC Web QA tool is not configured. "
            "Call configure_idc_web_qa_tool() at startup."
        )

    task = Task(user_msg=question, files=[], kwargs={})
    return await _IDC_QA.run(task, _cs(), top_k=top_k, max_chars=max_chars)
