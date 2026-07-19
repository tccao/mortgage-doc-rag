"""Index construction and retrieval over processed chunks."""

from __future__ import annotations

from .chunking import ChunkMetadata
from .config import PipelineConfig

_EMBED_CONFIGURED = False


def configure_embeddings(cfg: PipelineConfig) -> None:
    """Set the global LlamaIndex embedding model once (no LLM attached)."""
    global _EMBED_CONFIGURED
    if _EMBED_CONFIGURED:
        return
    from llama_index.core import Settings
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    Settings.embed_model = HuggingFaceEmbedding(model_name=cfg.embed_model)
    Settings.llm = None
    _EMBED_CONFIGURED = True


def build_index(chunks: list[ChunkMetadata], filename: str, cfg: PipelineConfig):
    """Build a VectorStoreIndex from chunks, preserving provenance metadata."""
    from llama_index.core import Document, VectorStoreIndex

    configure_embeddings(cfg)
    documents = [
        Document(
            text=chunk.text,
            metadata={
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "type": chunk.doc_type,
                "page_start": chunk.page_start + 1,
                "page_end": chunk.page_end + 1,
                "chunk_index": chunk.chunk_index,
                "filename": filename,
            },
        )
        for chunk in chunks
    ]
    return VectorStoreIndex.from_documents(documents)


def build_reranker(cfg: PipelineConfig):
    from llama_index.core.postprocessor import SentenceTransformerRerank

    return SentenceTransformerRerank(model=cfg.rerank_model, top_n=cfg.rerank_top_n)


def retrieve(index, question: str, cfg: PipelineConfig, doc_type: str | None = None, reranker=None):
    """Top-k retrieval with optional doc-type metadata filter and reranking."""
    filters = None
    if doc_type and doc_type != "All":
        from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters

        filters = MetadataFilters(filters=[ExactMatchFilter(key="type", value=doc_type)])

    retriever = index.as_retriever(similarity_top_k=cfg.top_k, filters=filters)
    nodes = retriever.retrieve(question)

    if reranker is not None and nodes:
        from llama_index.core import QueryBundle

        nodes = reranker.postprocess_nodes(nodes, query_bundle=QueryBundle(question))
    return nodes
