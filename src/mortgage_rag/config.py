"""Single configuration object for the whole pipeline.

One ``PipelineConfig`` drives OCR, chunking, retrieval, the LLM backend, and the
orchestrator mode. v1 (classical) and v2 (agentic) differ only in ``mode`` so the
two never grow separate config trees.

Precedence: explicit kwargs > environment variables (``MRAG_*``) > config.yaml > defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

import yaml

_ENV_PREFIX = "MRAG_"


@dataclass
class PipelineConfig:
    # Orchestrator
    mode: Literal["classical", "agentic"] = "classical"

    # LLM backend
    llm_backend: Literal["llama_cpp", "openai_compat", "mock"] = "llama_cpp"
    llm_repo_id: str = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
    llm_filename: str = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
    llm_api_base: str = ""  # openai_compat only
    llm_api_key: str = ""  # openai_compat only; prefer MRAG_LLM_API_KEY env var
    llm_model_name: str = ""  # openai_compat only
    temperature: float = 0.0
    max_new_tokens: int = 512
    context_window: int = 4096
    n_gpu_layers: int = -1  # -1 = all layers on GPU when available, 0 = CPU

    # Embeddings / retrieval
    embed_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Off by default as of ADR-10: scored at equal depth, the cross-encoder
    # demoted gold documents (MRR 0.862 -> 0.828, 3 golds dropped) and caused two
    # answer failures. Re-enable per-arm with --set use_reranker=true.
    use_reranker: bool = False
    top_k: int = 5
    rerank_top_n: int = 3

    # Classification
    # IDF-weighted keyword scoring instead of equal-weight counting. Off by
    # default so the committed baseline stays comparable; flip it as an ablation
    # arm (--set use_idf_classifier=true) to measure what the weighting buys.
    use_idf_classifier: bool = False

    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 100
    use_semantic_chunking: bool = False

    # OCR
    ocr_dpi: int = 300
    min_digital_chars: int = 50  # below this, a page is treated as scanned

    # Determinism
    seed: int = 42

    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, yaml_path: str | Path | None = None, **overrides: Any) -> "PipelineConfig":
        values: dict[str, Any] = {}

        if yaml_path is None and Path("config.yaml").exists():
            yaml_path = "config.yaml"
        if yaml_path is not None and Path(yaml_path).exists():
            with open(yaml_path) as f:
                values.update(yaml.safe_load(f) or {})

        for f_ in fields(cls):
            env_val = os.environ.get(_ENV_PREFIX + f_.name.upper())
            if env_val is not None:
                if f_.type in ("int", int):
                    values[f_.name] = int(env_val)
                elif f_.type in ("float", float):
                    values[f_.name] = float(env_val)
                elif f_.type in ("bool", bool):
                    values[f_.name] = env_val.lower() in ("1", "true", "yes")
                else:
                    values[f_.name] = env_val

        values.update({k: v for k, v in overrides.items() if v is not None})
        known = {f_.name for f_ in fields(cls)}
        extra = {k: v for k, v in values.items() if k not in known}
        values = {k: v for k, v in values.items() if k in known}
        cfg = cls(**values)
        cfg.extra.update(extra)
        return cfg
