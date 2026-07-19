"""LLM backend abstraction.

``LLMBackend`` is the single seam between the pipeline and any language model.
Swapping models (Mistral GGUF -> Ornith GGUF -> hosted endpoint) is one config
field; nothing above this layer changes.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .config import PipelineConfig


@runtime_checkable
class LLMBackend(Protocol):
    def complete(self, prompt: str) -> str:
        """Return the model's completion for a single prompt."""
        ...


class LlamaCppBackend:
    """Local GGUF model via llama-cpp-python, downloaded from the HF Hub."""

    def __init__(self, cfg: PipelineConfig):
        import torch
        from huggingface_hub import hf_hub_download
        from llama_index.llms.llama_cpp import LlamaCPP

        model_path = hf_hub_download(repo_id=cfg.llm_repo_id, filename=cfg.llm_filename)
        n_gpu_layers = cfg.n_gpu_layers if torch.cuda.is_available() else 0
        self._llm = LlamaCPP(
            model_path=model_path,
            temperature=cfg.temperature,
            max_new_tokens=cfg.max_new_tokens,
            context_window=cfg.context_window,
            model_kwargs={"n_gpu_layers": n_gpu_layers, "verbose": False},
            verbose=False,
        )

    def complete(self, prompt: str) -> str:
        return self._llm.complete(prompt).text.strip()


class OpenAICompatBackend:
    """Any OpenAI-compatible endpoint: llama-server, vLLM, or a hosted API."""

    def __init__(self, cfg: PipelineConfig):
        import requests

        self._session = requests.Session()
        self._base = cfg.llm_api_base.rstrip("/")
        self._model = cfg.llm_model_name
        self._temperature = cfg.temperature
        self._max_tokens = cfg.max_new_tokens
        if cfg.llm_api_key:
            self._session.headers["Authorization"] = f"Bearer {cfg.llm_api_key}"

    def complete(self, prompt: str) -> str:
        resp = self._session.post(
            f"{self._base}/chat/completions",
            json={
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self._temperature,
                "max_tokens": self._max_tokens,
            },
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


class MockBackend:
    """Deterministic stub for tests and no-LLM CI runs."""

    def __init__(self, cfg: PipelineConfig | None = None, canned: str = "MOCK_ANSWER"):
        self.canned = canned
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.canned


def build_backend(cfg: PipelineConfig) -> LLMBackend:
    if cfg.llm_backend == "llama_cpp":
        return LlamaCppBackend(cfg)
    if cfg.llm_backend == "openai_compat":
        return OpenAICompatBackend(cfg)
    if cfg.llm_backend == "mock":
        return MockBackend(cfg)
    raise ValueError(f"Unknown llm_backend: {cfg.llm_backend}")
