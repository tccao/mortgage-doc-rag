"""End-to-end ingestion: PDFs -> pages -> logical documents -> chunks -> index.

Pure functions returning a ``PipelineResult``; no globals, no UI coupling.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any

from .chunking import ChunkMetadata, LogicalDocument, PageInfo, run_advanced_pipeline
from .config import PipelineConfig
from .validation import cross_check_consistency, validate_extracted_data


@dataclass
class ProcessedFile:
    filename: str
    pages: list[PageInfo]
    logical_docs: list[LogicalDocument]
    chunks: list[ChunkMetadata]
    validation_data: dict[str, Any]
    validation_issues: list[str]


@dataclass
class PipelineResult:
    files: list[ProcessedFile] = field(default_factory=list)
    index: Any = None
    consistency_report: str = ""
    errors: list[str] = field(default_factory=list)

    @property
    def stats(self) -> dict[str, Any]:
        type_counts: dict[str, int] = {}
        for f in self.files:
            for ld in f.logical_docs:
                type_counts[ld.doc_type] = type_counts.get(ld.doc_type, 0) + 1
        return {
            "files_processed": len(self.files),
            "total_pages": sum(len(f.pages) for f in self.files),
            "total_documents": sum(len(f.logical_docs) for f in self.files),
            "total_chunks": sum(len(f.chunks) for f in self.files),
            "doc_type_breakdown": type_counts,
            "errors": len(self.errors),
        }


def process_files(
    pdf_paths: list[str], cfg: PipelineConfig | None = None, build_vector_index: bool = True
) -> PipelineResult:
    """Process PDFs through OCR, doc separation, validation, and (optionally) indexing."""
    cfg = cfg or PipelineConfig.load()
    result = PipelineResult()
    all_chunks: list[tuple[str, ChunkMetadata]] = []
    knowledge = []

    for path in pdf_paths:
        fname = os.path.basename(path)
        try:
            pages, logical_docs, chunks = run_advanced_pipeline(
                path,
                use_semantic_chunking=cfg.use_semantic_chunking,
                chunk_size=cfg.chunk_size,
                overlap=cfg.chunk_overlap,
            )
            full_text = "\n\n".join(p.text for p in pages)
            data, issues = validate_extracted_data(full_text, fname)
            knowledge.append({"filename": fname, "data": data})

            result.files.append(
                ProcessedFile(
                    filename=fname,
                    pages=pages,
                    logical_docs=logical_docs,
                    chunks=chunks,
                    validation_data=data,
                    validation_issues=issues,
                )
            )
            all_chunks.extend((fname, c) for c in chunks)
        except Exception as e:
            result.errors.append(f"{fname}: {e}")

    result.consistency_report = cross_check_consistency(knowledge)

    if build_vector_index and all_chunks:
        # One index across all files; chunk metadata keeps per-file provenance.
        from llama_index.core import Document, VectorStoreIndex

        from .rag import configure_embeddings

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
                    "filename": fname_,
                },
            )
            for fname_, chunk in all_chunks
        ]
        result.index = VectorStoreIndex.from_documents(documents)

    return result


def main() -> None:
    """CLI: uv run python -m mortgage_rag.pipeline <pdf> [pdf ...]"""
    paths = sys.argv[1:]
    if not paths:
        print("usage: python -m mortgage_rag.pipeline <pdf> [pdf ...]")
        raise SystemExit(1)

    cfg = PipelineConfig.load()
    result = process_files(paths, cfg, build_vector_index=False)

    for f in result.files:
        print(f"\n{f.filename}: {len(f.pages)} pages, {len(f.logical_docs)} documents")
        for ld in f.logical_docs:
            print(f"  - {ld.doc_type} (pages {ld.page_start + 1}-{ld.page_end + 1}, "
                  f"confidence {ld.confidence:.0%}, {len(ld.chunks)} chunks)")
        for issue in f.validation_issues:
            print(f"  ! {issue}")

    print(f"\n{result.consistency_report}")
    print(f"\nStats: {result.stats}")
    if result.errors:
        print("\nErrors:")
        for e in result.errors:
            print(f"  ! {e}")


if __name__ == "__main__":
    main()
