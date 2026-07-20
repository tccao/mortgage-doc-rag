"""Multi-document PDF processing: page extraction, doc-type classification,
boundary detection, logical-document grouping, and metadata-preserving chunking.

A single uploaded PDF often bundles several real documents (loan estimate,
pay slips, W-2, ...). Chunks carry doc_type + page provenance so retrieval can
filter by type and evals can score against gold pages.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import fitz

from .ocr import extract_page_texts


@dataclass
class PageInfo:
    page_num: int
    text: str
    doc_type: str | None = None
    page_in_doc: int = 0
    word_count: int = 0
    has_tables: bool = False
    has_images: bool = False


@dataclass
class LogicalDocument:
    doc_id: str
    doc_type: str
    page_start: int
    page_end: int
    text: str
    chunks: list = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class ChunkMetadata:
    chunk_id: str
    doc_id: str
    doc_type: str
    chunk_index: int
    page_start: int
    page_end: int
    text: str
    word_count: int = 0


DOC_TYPE_PATTERNS: dict[str, list[str]] = {
    "Loan Estimate": ["loan estimate", "estimated closing costs", "rate lock", "loan costs",
                      "estimated cash to close", "services you can shop for"],
    "Closing Disclosure": ["closing disclosure", "closing costs", "cash to close",
                           "loan disclosures", "projected payments", "escrow account"],
    "Settlement Statement": ["settlement statement", "hud-1", "settlement charges",
                             "settlement agent", "gross amount due"],
    "Loan Application": ["uniform residential loan application", "borrower information",
                         "1003", "urla", "declarations", "assets and liabilities"],
    "Appraisal": ["appraisal report", "appraised value", "subject property", "comparable sales",
                  "uniform residential appraisal"],
    "Lender Fee Sheet": ["origination", "lender fee", "title fee", "broker fee", "loan fee"],
    "Pay Slip": ["gross pay", "net pay", "deductions", "earnings", "pay period", "ytd",
                 "withholding", "salary", "wage"],
    "Resume": ["experience", "education", "skills", "objective", "employment", "references",
               "qualifications"],
    "Mortgage Contract": ["promissory note", "deed of trust", "loan amount", "principal",
                          "interest rate", "mortgage"],
    "Bank Statement": ["account number", "balance", "transactions", "deposits", "withdrawals",
                       "statement period"],
    "Tax Document": ["w-2", "w2", "1099", "1098", "tax return", "irs", "federal tax",
                     "adjusted gross", "taxable income", "form 1040", "4506"],
    "Invoice": ["invoice", "bill to", "amount due", "payment terms", "invoice number",
                "subtotal"],
    "Contract": ["agreement", "parties", "terms and conditions", "hereby", "whereas",
                 "obligations"],
}

# Phrases that signal the start of a new document inside a multi-doc PDF.
BOUNDARY_MARKERS = [
    "payslip", "pay slip", "pay date", "employee name", "employee id",
    "invoice #", "invoice number", "invoice date",
    "loan estimate", "closing disclosure", "settlement statement",
    "uniform residential loan application",
    "page 1 of", "page 1/", "statement period", "pay period",
]


def compute_keyword_idf(texts: list[str]) -> dict[str, float]:
    """Inverse document frequency for every classifier keyword, over a corpus.

    Raw keyword counting assumes every keyword carries equal evidence. It does
    not: terms like "mortgage", "principal", and "interest rate" appear on nearly
    every document in a loan file, so any doc type listing them scores for free
    and becomes a sink that absorbs unrelated pages. Weighting by

        idf(term) = log(N / df(term))

    drives a term present in every document to ~0 and leaves the discriminative
    phrases carrying the decision. Same principle as BM25's idf factor.
    """
    n = len(texts)
    if not n:
        return {}
    lowered = [t.lower() for t in texts]
    keywords = {kw for kws in DOC_TYPE_PATTERNS.values() for kw in kws}
    idf = {}
    for kw in keywords:
        df = sum(1 for t in lowered if kw in t)
        idf[kw] = math.log(n / df) if df else math.log(n)
    return idf


def classify_page_content(
    text: str, idf: dict[str, float] | None = None
) -> tuple[str, float]:
    """Keyword-scored doc-type classification. Returns (doc_type, confidence 0-1).

    With ``idf`` supplied, keyword hits are weighted by corpus inverse document
    frequency instead of counted equally. Off by default: the committed baseline
    was measured with plain counting, and switching the default would change the
    reported number without a re-run to back it.
    """
    text_lower = text.lower()
    scores = {}
    for doc_type, keywords in DOC_TYPE_PATTERNS.items():
        if idf:
            score = sum(idf.get(kw, 0.0) for kw in keywords if kw in text_lower)
            # Normalize by the type's own maximum so a doc type listing many rare
            # keywords cannot outscore one listing few, purely on list length.
            ceiling = sum(idf.get(kw, 0.0) for kw in keywords) or 1.0
            score = score / ceiling * len(keywords)
        else:
            score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[doc_type] = score

    if scores:
        best_type = max(scores, key=scores.get)
        confidence = min(scores[best_type] / 5.0, 1.0)
        return best_type, confidence
    return "Other", 0.0


def extract_pages_from_pdf(pdf_path: str, dpi: int = 200) -> list[PageInfo]:
    texts = extract_page_texts(pdf_path, dpi=dpi)
    pages = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            text = texts[i]
            has_tables = "|" in text or text.count("\t") > 5
            pages.append(
                PageInfo(
                    page_num=i,
                    text=text,
                    word_count=len(text.split()),
                    has_tables=has_tables,
                    has_images=len(page.get_images()) > 0,
                )
            )
    return pages


def detect_boundary(prev_text: str, curr_text: str, curr_type: str) -> bool:
    """True if the current page continues the same document, False if a new one starts."""
    if not prev_text:
        return True

    new_type, _ = classify_page_content(curr_text)
    if new_type != curr_type and new_type != "Other":
        return False

    curr_start = curr_text[:500].lower()
    return not any(marker in curr_start for marker in BOUNDARY_MARKERS)


def group_pages_into_documents(pages: list[PageInfo]) -> list[LogicalDocument]:
    logical_docs: list[LogicalDocument] = []
    current_pages: list[PageInfo] = []
    current_type = "Other"
    current_confidence = 0.0
    doc_counter = 0

    def flush():
        nonlocal doc_counter
        if current_pages:
            logical_docs.append(
                LogicalDocument(
                    doc_id=f"doc_{doc_counter}",
                    doc_type=current_type,
                    page_start=current_pages[0].page_num,
                    page_end=current_pages[-1].page_num,
                    text="\n\n".join(p.text for p in current_pages),
                    confidence=current_confidence,
                )
            )
            doc_counter += 1

    for i, page in enumerate(pages):
        if i == 0 or not detect_boundary(pages[i - 1].text, page.text, current_type):
            if i != 0:
                flush()
            current_type, current_confidence = classify_page_content(page.text)
            page.doc_type = current_type
            current_pages = [page]
        else:
            page.doc_type = current_type
            current_pages.append(page)

    flush()
    return logical_docs


def chunk_document_with_metadata(
    logical_doc: LogicalDocument, chunk_size: int = 500, overlap: int = 100
) -> list[ChunkMetadata]:
    """Sliding-window word chunking with approximate page attribution."""
    chunks = []
    words = logical_doc.text.split()

    if len(words) <= chunk_size:
        chunks.append(
            ChunkMetadata(
                chunk_id=f"{logical_doc.doc_id}_chunk_0",
                doc_id=logical_doc.doc_id,
                doc_type=logical_doc.doc_type,
                chunk_index=0,
                page_start=logical_doc.page_start,
                page_end=logical_doc.page_end,
                text=logical_doc.text,
                word_count=len(words),
            )
        )
        return chunks

    stride = chunk_size - overlap
    for i, start_idx in enumerate(range(0, len(words), stride)):
        end_idx = min(start_idx + chunk_size, len(words))
        chunk_text = " ".join(words[start_idx:end_idx])

        chunk_position = start_idx / len(words)
        page_range = logical_doc.page_end - logical_doc.page_start
        relative_page = int(chunk_position * page_range)
        chunk_page_start = logical_doc.page_start + relative_page
        chunk_page_end = min(chunk_page_start + 1, logical_doc.page_end)

        chunks.append(
            ChunkMetadata(
                chunk_id=f"{logical_doc.doc_id}_chunk_{i}",
                doc_id=logical_doc.doc_id,
                doc_type=logical_doc.doc_type,
                chunk_index=i,
                page_start=chunk_page_start,
                page_end=chunk_page_end,
                text=chunk_text,
                word_count=len(chunk_text.split()),
            )
        )
        if end_idx >= len(words):
            break

    return chunks


def chunk_with_sentence_splitter(
    logical_doc: LogicalDocument, chunk_size: int = 500, chunk_overlap: int = 100
) -> list[ChunkMetadata]:
    """Semantic chunking via LlamaIndex SentenceSplitter."""
    from llama_index.core import Document
    from llama_index.core.node_parser import SentenceSplitter

    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        paragraph_separator="\n\n",
        separator=" ",
    )
    doc = Document(text=logical_doc.text)
    nodes = splitter.get_nodes_from_documents([doc])

    return [
        ChunkMetadata(
            chunk_id=f"{logical_doc.doc_id}_chunk_{i}",
            doc_id=logical_doc.doc_id,
            doc_type=logical_doc.doc_type,
            chunk_index=i,
            page_start=logical_doc.page_start,
            page_end=logical_doc.page_end,
            text=node.text,
            word_count=len(node.text.split()),
        )
        for i, node in enumerate(nodes)
    ]


def process_all_documents(
    logical_docs: list[LogicalDocument],
    use_semantic_chunking: bool = False,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[ChunkMetadata]:
    all_chunks = []
    for ld in logical_docs:
        if use_semantic_chunking:
            chunks = chunk_with_sentence_splitter(ld, chunk_size, overlap)
        else:
            chunks = chunk_document_with_metadata(ld, chunk_size, overlap)
        ld.chunks = chunks
        all_chunks.extend(chunks)
    return all_chunks


def run_advanced_pipeline(
    pdf_path: str,
    use_semantic_chunking: bool = False,
    chunk_size: int = 500,
    overlap: int = 100,
) -> tuple[list[PageInfo], list[LogicalDocument], list[ChunkMetadata]]:
    """Full processing for one PDF: pages -> logical documents -> chunks."""
    pages = extract_pages_from_pdf(pdf_path)
    logical_docs = group_pages_into_documents(pages)
    chunks = process_all_documents(logical_docs, use_semantic_chunking, chunk_size, overlap)
    return pages, logical_docs, chunks
