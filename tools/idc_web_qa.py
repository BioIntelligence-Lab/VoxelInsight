from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from core.state import Task, TaskResult, ConversationState
from core.llm_provider import choose_llm, OpenAISettings
from tools.shared import toolify_agent, _cs

# --- LlamaIndex imports (LOAD ONLY, NO BUILDING) ---
from llama_index.core import (
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.embeddings.openai import OpenAIEmbedding


# ============================================================
# Path + Index Loader Helpers
# ============================================================

def _get_index_dir() -> Path:
    """
    Resolve path for the prebuilt LlamaIndex directory:
    VoxelInsight/idc_index
    """
    here = Path(__file__).resolve().parent
    repo_root = here.parent  # e.g., /Users/vparekh/Research/VoxelInsight

    index_dir = (repo_root / "idc_index").resolve()

    print(f"[IDC RAG] Repo root:       {repo_root}")
    print(f"[IDC RAG] Index dir path:  {index_dir}")

    return index_dir


def _init_llamaindex_settings():
    """
    Configure LlamaIndex to use OpenAI embeddings at query time.
    Generation uses your existing OpenAI wrapper.
    """
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")


def _load_existing_index():
    """
    Load a pre-built LlamaIndex from VoxelInsight/idc_index.

    Assumes you already ran the external builder script
    (e.g., create_idc_web_llama_index.py) which:
      - recursively ingests multiple .md files
      - persists the index to idc_index/
    """
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
    """
    Concatenate retrieved node texts into a single context string,
    with an overall character cap so we never blow the context window.
    """
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


# ============================================================
# IDC Web QA Agent (mirrors DataQueryAgent pattern)
# ============================================================

class IDCWebQAAgent:
    name = "idc_web_qa"
    model = "gpt-5-nano"

    def __init__(self, system_prompt: Optional[str] = None):
        # 1. Load prebuilt index and create retriever
        index = _load_existing_index()
        self.retriever = index.as_retriever(similarity_top_k=5)

        # 2. LLM (reuse your provider)
        try:
            self.llm = choose_llm(
                openai_settings=OpenAISettings(model=self.model)
            )
        except Exception:
            self.llm = None

        # 3. System prompt
        self.system_prompt = system_prompt or (
            "You are an expert on the NCI Imaging Data Commons (IDC). "
            "You answer strictly from the provided IDC documentation context. "
            "If the answer is not covered in the context, say you do not know "
            "and optionally suggest where in the IDC docs the user might look."
        )

    async def run(
        self,
        task: Task,
        state: ConversationState,
        reasoning_effort: str = "low",
    ) -> TaskResult:
        if self.llm is None:
            raise RuntimeError("LLM provider is not configured.")

        question = task.user_msg

        # 1. Retrieve relevant chunks from the index
        nodes = self.retriever.retrieve(question)

        # 🔍 Optional debug: uncomment to inspect retrieved sources
        # for n in nodes:
        #     print("[IDC RAG HIT]", n.metadata.get("file_path"))

        context = _nodes_to_context(nodes)

        # 2. Build messages for your LLM
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Context (retrieved from local IDC docs snapshot):\n{context}"
                ),
            },
        ]

        # 3. Call your LLM wrapper
        answer = await self.llm.ainvoke(
            messages,
            reasoning_effort=reasoning_effort,
        )

        return TaskResult(
            output=answer,
            artifacts={
                "index_dir": "idc_index",
                "num_retrieved_chunks": len(nodes),
            },
        )


# ============================================================
# Tool Wrapper (Same Pattern as idc_query)
# ============================================================

_IDC_QA: Optional[IDCWebQAAgent] = None


def configure_idc_web_qa_tool(*, system_prompt: Optional[str] = None):
    """
    Configure the IDC Web QA tool once (similar to configure_idc_query_tool).
    Call this at startup.
    """
    global _IDC_QA
    _IDC_QA = IDCWebQAAgent(system_prompt=system_prompt)


class IDCWebQAArgs(BaseModel):
    question: str = Field(..., description="Natural language question about IDC documentation.")
    reasoning_effort: str = Field(
        "low",
        description=(
            "Reasoning effort level: 'minimal', 'low', 'medium'. "
            "Lower is faster; increase if answers are insufficient."
        ),
    )


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
    reasoning_effort: str = "low",
):
    if _IDC_QA is None:
        raise RuntimeError(
            "IDC Web QA tool is not configured. "
            "Call configure_idc_web_qa_tool() at startup."
        )

    task = Task(user_msg=question, files=[], kwargs={})
    return await _IDC_QA.run(task, _cs(), reasoning_effort=reasoning_effort)
