"""Orchestrator interface: the single entry point evals and the UI call.

``ClassicalRAG`` (v1) is retrieve -> rerank -> prompt -> generate.
``AgenticRAG`` (v2) will implement the same interface with a router + tool loop;
callers flip ``config.mode`` and nothing else changes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from .backends import LLMBackend, build_backend
from .config import PipelineConfig
from .rag import build_reranker, retrieve


@dataclass
class Citation:
    filename: str
    doc_type: str
    page_start: int
    page_end: int
    score: float
    chunk_id: str = ""


@dataclass
class Result:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)


class Orchestrator(Protocol):
    def answer(self, question: str, doc_type: str | None = None) -> Result: ...


QA_PROMPT = (
    "[INST] Answer briefly using only the context below. "
    "If the context does not contain the answer, say so.\n"
    "Context:\n{context}\n\nQuestion: {question} [/INST]"
)


class ClassicalRAG:
    """v1 pipeline: dense retrieval + optional cross-encoder rerank + single generation."""

    def __init__(self, index, cfg: PipelineConfig, backend: LLMBackend | None = None):
        self.index = index
        self.cfg = cfg
        self.backend = backend or build_backend(cfg)
        self.reranker = build_reranker(cfg) if cfg.use_reranker else None

    def answer(self, question: str, doc_type: str | None = None) -> Result:
        t0 = time.time()
        nodes = retrieve(self.index, question, self.cfg, doc_type=doc_type, reranker=self.reranker)
        t_retrieve = time.time() - t0

        if not nodes:
            return Result(
                answer="No relevant information found.",
                trace={"retrieval_s": t_retrieve, "generation_s": 0.0, "n_nodes": 0},
            )

        context = "\n\n".join(
            f"[{n.metadata.get('type', '?')}, pages "
            f"{n.metadata.get('page_start', '?')}-{n.metadata.get('page_end', '?')}]:\n{n.text}"
            for n in nodes[: self.cfg.rerank_top_n]
        )
        prompt = QA_PROMPT.format(context=context, question=question)

        t1 = time.time()
        answer = self.backend.complete(prompt)
        t_generate = time.time() - t1

        citations = [
            Citation(
                filename=n.metadata.get("filename", "?"),
                doc_type=n.metadata.get("type", "?"),
                page_start=n.metadata.get("page_start", 0),
                page_end=n.metadata.get("page_end", 0),
                score=float(getattr(n, "score", 0.0) or 0.0),
                chunk_id=n.metadata.get("chunk_id", ""),
            )
            for n in nodes
        ]
        return Result(
            answer=answer,
            citations=citations,
            trace={
                "mode": "classical",
                "retrieval_s": round(t_retrieve, 3),
                "generation_s": round(t_generate, 3),
                "n_nodes": len(nodes),
                "prompt_chars": len(prompt),
            },
        )


def build_orchestrator(index, cfg: PipelineConfig, backend: LLMBackend | None = None) -> Orchestrator:
    if cfg.mode == "classical":
        return ClassicalRAG(index, cfg, backend)
    if cfg.mode == "agentic":
        from .agent import AgenticRAG  # v2, lands in a follow-up PR

        return AgenticRAG(index, cfg, backend)
    raise ValueError(f"Unknown mode: {cfg.mode}")
