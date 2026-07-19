"""No-LLM smoke tests: config precedence, validation rules, chunking/classification,
and the orchestrator against a stub index with a mock backend. All CPU, CI-safe."""


from mortgage_rag.backends import MockBackend
from mortgage_rag.chunking import (
    ChunkMetadata,
    LogicalDocument,
    chunk_document_with_metadata,
    classify_page_content,
    detect_boundary,
    group_pages_into_documents,
    PageInfo,
)
from mortgage_rag.config import PipelineConfig
from mortgage_rag.validation import cross_check_consistency, validate_extracted_data


def test_config_defaults_and_env(monkeypatch):
    cfg = PipelineConfig.load()
    assert cfg.mode == "classical"
    assert cfg.temperature == 0.0

    monkeypatch.setenv("MRAG_TOP_K", "7")
    monkeypatch.setenv("MRAG_MODE", "agentic")
    cfg = PipelineConfig.load()
    assert cfg.top_k == 7
    assert cfg.mode == "agentic"


def test_validation_loan_amount_and_ssn():
    text = "Loan Amount: $250,000.00 at 6.875% closing fee $1,200.50 date 01/15/2026"
    data, issues = validate_extracted_data(text)
    assert data["loan_amount"] == 250000.00
    assert "6.875" in data["interest_rate"]
    assert not any("SSN" in i for i in issues)

    _, issues = validate_extracted_data("SSN: 123-45-6789")
    assert any("SSN" in i for i in issues)


def test_cross_check_detects_mismatch():
    consistent = [
        {"filename": "a.pdf", "data": {"loan_amount": 100000.0}},
        {"filename": "b.pdf", "data": {"loan_amount": 100000.0}},
    ]
    assert "consistent" in cross_check_consistency(consistent).lower()

    mismatched = consistent + [{"filename": "c.pdf", "data": {"loan_amount": 1.0}}]
    assert "mismatch" in cross_check_consistency(mismatched).lower()


def test_classification():
    doc_type, conf = classify_page_content(
        "Closing Disclosure. This form is a statement of final loan terms and closing costs. "
        "Projected payments, cash to close, loan disclosures."
    )
    assert doc_type == "Closing Disclosure"
    assert conf > 0

    doc_type, _ = classify_page_content("gross pay net pay deductions pay period ytd")
    assert doc_type == "Pay Slip"


def test_boundary_detection_and_grouping():
    payslip = "Pay period 01/01-01/15. Gross pay $5,000 net pay $3,800 deductions ytd"
    resume = "Experience: engineer. Education: BS. Skills: Python. References available."
    pages = [
        PageInfo(page_num=0, text=payslip),
        PageInfo(page_num=1, text=resume),
    ]
    docs = group_pages_into_documents(pages)
    assert len(docs) == 2
    assert docs[0].doc_type == "Pay Slip"
    assert docs[1].doc_type == "Resume"

    assert detect_boundary("", "anything", "Other") is True


def test_chunking_metadata_preserved():
    ld = LogicalDocument(
        doc_id="doc_0",
        doc_type="Pay Slip",
        page_start=2,
        page_end=4,
        text=" ".join(f"word{i}" for i in range(1200)),
    )
    chunks = chunk_document_with_metadata(ld, chunk_size=500, overlap=100)
    assert len(chunks) >= 3
    assert all(isinstance(c, ChunkMetadata) for c in chunks)
    assert all(c.doc_type == "Pay Slip" for c in chunks)
    assert chunks[0].page_start == 2
    assert chunks[-1].page_end <= 4


def test_mock_backend_records_prompts():
    backend = MockBackend(canned="42")
    assert backend.complete("[INST] q [/INST]") == "42"
    assert backend.prompts == ["[INST] q [/INST]"]


class _StubNode:
    def __init__(self, text, meta, score=0.9):
        self.text = text
        self.metadata = meta
        self.score = score


class _StubRetriever:
    def __init__(self, nodes):
        self._nodes = nodes

    def retrieve(self, _q):
        return self._nodes


class _StubIndex:
    def __init__(self, nodes):
        self._nodes = nodes

    def as_retriever(self, similarity_top_k=5, filters=None):
        return _StubRetriever(self._nodes[:similarity_top_k])


def test_classical_orchestrator_with_stub_index():
    from mortgage_rag.orchestrator import ClassicalRAG

    nodes = [
        _StubNode(
            "Loan amount is $250,000.00",
            {
                "type": "Closing Disclosure",
                "page_start": 1,
                "page_end": 1,
                "filename": "cd.pdf",
                "chunk_id": "doc_0_chunk_0",
            },
        )
    ]
    cfg = PipelineConfig(llm_backend="mock", use_reranker=False)
    orch = ClassicalRAG(_StubIndex(nodes), cfg, backend=MockBackend(canned="$250,000.00"))

    res = orch.answer("What is the loan amount?")
    assert res.answer == "$250,000.00"
    assert res.citations[0].doc_type == "Closing Disclosure"
    assert res.trace["mode"] == "classical"
    assert "Loan amount is $250,000.00" in orch.backend.prompts[0]


def test_orchestrator_empty_index():
    from mortgage_rag.orchestrator import ClassicalRAG

    cfg = PipelineConfig(llm_backend="mock", use_reranker=False)
    orch = ClassicalRAG(_StubIndex([]), cfg, backend=MockBackend())
    res = orch.answer("anything")
    assert "No relevant information" in res.answer
    assert res.citations == []
