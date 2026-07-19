"""Mortgage document RAG pipeline with a layered evaluation harness."""

from typing import Any

__version__ = "0.1.0"

__all__ = ["PipelineConfig", "Result", "build_orchestrator", "PipelineResult", "process_files"]

_EXPORTS = {
    "PipelineConfig": ("mortgage_rag.config", "PipelineConfig"),
    "Result": ("mortgage_rag.orchestrator", "Result"),
    "build_orchestrator": ("mortgage_rag.orchestrator", "build_orchestrator"),
    "PipelineResult": ("mortgage_rag.pipeline", "PipelineResult"),
    "process_files": ("mortgage_rag.pipeline", "process_files"),
}


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        import importlib

        module, attr = _EXPORTS[name]
        return getattr(importlib.import_module(module), attr)
    raise AttributeError(f"module 'mortgage_rag' has no attribute {name!r}")
