from __future__ import annotations

from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field

from core.state import Task, TaskResult, ConversationState
from core.llm_provider import choose_llm
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

    def __init__(self, system_prompt: str):
        index = _load_existing_index()
        self.index = index
        self.system_prompt = system_prompt
        try:
            self.llm = choose_llm()
        except Exception:
            self.llm = None

    async def run(
        self,
        task: Task,
        state: ConversationState,
        top_k: int = 5,
        max_chars: int = 8000,
        synthesize: bool = True,
        reasoning_effort: str = "low",
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
            "note": "Synthesis is optional; when enabled, the answer field contains a grounded response.",
        }

        if synthesize and self.llm is not None:
            sources = sorted({c.get("source") for c in chunk_list if c.get("source")})
            sources_text = "\n".join(sources) if sources else "None"
            messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": f"Question:\n{question}\n\nContext:\n{context}\n\nSources:\n{sources_text}",
                },
            ]
            answer = await self.llm.ainvoke(messages, temperature=1, reasoning_effort=reasoning_effort)
            payload["answer"] = answer
            payload["sources"] = sources
            return TaskResult(output=payload, artifacts=payload)

        return TaskResult(output=payload, artifacts=payload)

_IDC_QA: Optional[IDCWebQAAgent] = None


def configure_idc_web_qa_tool(*, system_prompt: str):
    global _IDC_QA
    _IDC_QA = IDCWebQAAgent(system_prompt=system_prompt)


class IDCWebQAArgs(BaseModel):
    question: str = Field(..., description="Natural language question about IDC documentation.")
    top_k: int = Field(5, ge=1, le=20, description="How many chunks to retrieve from the IDC docs index.")
    max_chars: int = Field(8000, ge=500, le=20000, description="Max total characters to return across all retrieved chunks.")
    synthesize: bool = Field(True, description="Whether to synthesize a grounded answer from retrieved chunks.")
    reasoning_effort: str = Field("low", description="Reasoning effort level: 'minimal', 'low', 'medium'.")


@toolify_agent(
    name="idc_web_qa",
    description=(
        "Answers questions about the NCI Imaging Data Commons (IDC) using a local markdown knowledge base that has been pre-indexed with LlamaIndex and "
        "\nWhen the user asks questions about IDC documentation, use the `idc_web_qa` tool to answer them based. These are questions like \"What is the purpose of IDC?\", \"How to access IDC data?\", \"What collections are available in IDC?\", etc."
        "\nPrimarily if the user's question is a How to or what is, use the `idc_web_qa` tool to answer them based on IDC documentation."
        "\nYou may also want to use the idc_code_qa tool for how to questions to provide code examples."
        "\n- `idc_web_qa`: answer general IDC questions grounded in learn.canceridc.dev (or a provided IDC doc URL). Use when the user asks doc questions."
    ),
    args_schema=IDCWebQAArgs,
    timeout_s=60,
)
async def idc_web_qa_runner(
    question: str,
    top_k: int = 5,
    max_chars: int = 8000,
    synthesize: bool = True,
    reasoning_effort: str = "low",
):
    if _IDC_QA is None:
        raise RuntimeError(
            "IDC Web QA tool is not configured. "
            "Call configure_idc_web_qa_tool() at startup."
        )

    task = Task(user_msg=question, files=[], kwargs={})
    return await _IDC_QA.run(
        task,
        _cs(),
        top_k=top_k,
        max_chars=max_chars,
        synthesize=synthesize,
        reasoning_effort=reasoning_effort,
    )
