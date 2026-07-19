"""Gradio Q&A UI over the pipeline. Requires the `ui` and `llm` extras.

Run: uv run --extra ui --extra llm python -m mortgage_rag.app
"""

from __future__ import annotations

import gradio as gr

from .config import PipelineConfig
from .orchestrator import build_orchestrator
from .pipeline import process_files


class AppState:
    def __init__(self):
        self.cfg = PipelineConfig.load()
        self.result = None
        self.orchestrator = None


def create_app(state: AppState | None = None) -> gr.Blocks:
    state = state or AppState()

    def process_handler(files, use_semantic, chunk_size, chunk_overlap):
        if not files:
            return "Upload at least one PDF.", "", gr.update(choices=["All"])

        state.cfg.use_semantic_chunking = bool(use_semantic)
        state.cfg.chunk_size = int(chunk_size)
        state.cfg.chunk_overlap = int(chunk_overlap)

        paths = [f.name for f in files]
        state.result = process_files(paths, state.cfg)
        state.orchestrator = build_orchestrator(state.result.index, state.cfg)

        stats = state.result.stats
        status_lines = [
            f"**Processed {stats['files_processed']} file(s)** — "
            f"{stats['total_pages']} pages, {stats['total_documents']} documents, "
            f"{stats['total_chunks']} chunks",
            "",
            state.result.consistency_report,
        ]
        for f in state.result.files:
            for issue in f.validation_issues:
                status_lines.append(f"- ⚠️ {f.filename}: {issue}")
        for e in state.result.errors:
            status_lines.append(f"- ❌ {e}")

        structure_lines = []
        doc_types = set()
        for f in state.result.files:
            structure_lines.append(f"**{f.filename}**")
            for ld in f.logical_docs:
                doc_types.add(ld.doc_type)
                pg = (
                    f"{ld.page_start + 1}-{ld.page_end + 1}"
                    if ld.page_start != ld.page_end
                    else f"{ld.page_start + 1}"
                )
                structure_lines.append(
                    f"- {ld.doc_type} (pages {pg}, confidence {ld.confidence:.0%}, "
                    f"{len(ld.chunks)} chunks)"
                )

        return (
            "\n".join(status_lines),
            "\n".join(structure_lines),
            gr.update(choices=["All"] + sorted(doc_types), value="All"),
        )

    def chat(message, history, doc_filter):
        history = history + [{"role": "user", "content": message}]
        if state.orchestrator is None:
            return history + [{"role": "assistant", "content": "Process a PDF first."}]

        res = state.orchestrator.answer(message, doc_type=doc_filter)
        sources = "\n".join(
            f"- {c.doc_type} (pages {c.page_start}-{c.page_end}) — score {c.score:.3f}"
            for c in res.citations[:3]
        )
        reply = res.answer
        if sources:
            reply += f"\n\n**Sources:**\n{sources}"
        reply += (
            f"\n\n*retrieval {res.trace.get('retrieval_s', '?')}s | "
            f"generation {res.trace.get('generation_s', '?')}s*"
        )
        return history + [{"role": "assistant", "content": reply}]

    with gr.Blocks(title="Mortgage Doc RAG") as demo:
        gr.Markdown("# Mortgage Document Q&A")
        gr.Markdown("Dual-engine OCR → document separation → validated RAG")

        with gr.Row():
            with gr.Column(scale=2):
                pdf_input = gr.File(label="Upload PDFs", file_types=[".pdf"], file_count="multiple")
                with gr.Accordion("Advanced settings", open=False):
                    use_semantic = gr.Checkbox(False, label="Semantic chunking")
                    chunk_size = gr.Slider(100, 1000, 500, step=50, label="Chunk size (words)")
                    chunk_overlap = gr.Slider(0, 200, 100, step=25, label="Overlap (words)")
                process_btn = gr.Button("Process documents", variant="primary")

            with gr.Column(scale=1):
                status_out = gr.Markdown("Waiting for PDFs…")
                struct_out = gr.Markdown("")
                doc_filter = gr.Dropdown(["All"], value="All", label="Document type filter")

            with gr.Column(scale=2):
                chatbot = gr.Chatbot(height=420, type="messages")
                msg = gr.Textbox(placeholder="Ask a question…", show_label=False)

        process_btn.click(
            process_handler,
            [pdf_input, use_semantic, chunk_size, chunk_overlap],
            [status_out, struct_out, doc_filter],
        )
        msg.submit(chat, [msg, chatbot, doc_filter], [chatbot]).then(lambda: "", outputs=[msg])

    return demo


def main() -> None:
    create_app().launch()


if __name__ == "__main__":
    main()
