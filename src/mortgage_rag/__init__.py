"""Mortgage document RAG pipeline with a layered evaluation harness."""

from .config import PipelineConfig
from .orchestrator import Result, build_orchestrator
from .pipeline import PipelineResult, process_files

__all__ = ["PipelineConfig", "Result", "build_orchestrator", "PipelineResult", "process_files"]

__version__ = "0.1.0"
