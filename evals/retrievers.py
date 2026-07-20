"""Reference retrievers for baseline comparison.

The committed baselines in ``baselines/`` are *regression* baselines — the same
system's previous run, answering "did I break anything?". They cannot answer "is
this pipeline better than something simpler?". These are the reference baselines
that can:

  dense (default)  the shipped pipeline: bge-small bi-encoder over chunks
  bm25             classical sparse ranking — no embedding model, no GPU
  none             no retrieval at all; the LLM answers from parametric memory

``none`` matters more than it looks on this corpus. The documents are public
CFPB/IRS/HUD sample forms that are plausibly in the model's training data, so a
passing answer might be recall rather than retrieval. Without this baseline that
confound is unmeasured.

Self-contained by design: BM25 is ~40 lines and adding a retrieval dependency to
compare against a retrieval dependency defeats the purpose.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

BM25_K1 = 1.5  # term-frequency saturation: the 20th hit adds little over the 5th
BM25_B = 0.75  # length normalization strength (0 = off, 1 = full)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class ScoredNode:
    """Duck-types the LlamaIndex node fields the eval harness reads."""

    text: str
    metadata: dict = field(default_factory=dict)
    score: float = 0.0


class BM25Retriever:
    """Okapi BM25 over the same chunks the dense index is built from.

        score(q, d) = Σ  idf(t) · ( tf · (k1+1) ) / ( tf + k1·(1 − b + b·|d|/avgdl) )

    Two ideas worth stating in an interview: term frequency *saturates* (k1), so a
    chunk repeating a term 50 times does not beat one with 5; and length is
    normalized (b), so long chunks do not win by size alone. The idf term is the
    same principle that fixes the keyword classifier's over-firing class — terms
    appearing in every document carry no evidence.

    Expected to be competitive here: mortgage forms use fixed vocabulary
    ("Loan Amount", "Annual Percentage Rate") that appears verbatim in both the
    question and the document, which is exactly BM25's strong regime. If it ties
    dense retrieval, that is a finding worth reporting, not an embarrassment.
    """

    def __init__(self, chunks: list[tuple[str, object]], k1: float = BM25_K1, b: float = BM25_B):
        self.k1, self.b = k1, b
        self.nodes: list[ScoredNode] = []
        self.tokenized: list[Counter] = []
        self.lengths: list[int] = []

        for filename, chunk in chunks:
            self.nodes.append(
                ScoredNode(
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
            )
            tokens = tokenize(chunk.text)
            self.tokenized.append(Counter(tokens))
            self.lengths.append(len(tokens))

        n = len(self.nodes)
        self.avgdl = (sum(self.lengths) / n) if n else 0.0

        df: Counter = Counter()
        for counts in self.tokenized:
            df.update(counts.keys())
        # Robertson/Sparck-Jones idf with +0.5 smoothing; max() floors the value so
        # a term present in every chunk contributes ~0 rather than going negative.
        self.idf = {
            term: max(1e-9, math.log(1 + (n - freq + 0.5) / (freq + 0.5)))
            for term, freq in df.items()
        }

    def retrieve(self, query: str, top_k: int = 5) -> list[ScoredNode]:
        terms = tokenize(query)
        scored: list[tuple[float, int]] = []
        for i, counts in enumerate(self.tokenized):
            norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / (self.avgdl or 1))
            total = 0.0
            for term in terms:
                tf = counts.get(term, 0)
                if tf:
                    total += self.idf.get(term, 0.0) * (tf * (self.k1 + 1)) / (tf + norm)
            if total > 0:
                scored.append((total, i))

        scored.sort(key=lambda x: (-x[0], x[1]))  # stable: ties break by chunk order
        out = []
        for score, i in scored[:top_k]:
            node = self.nodes[i]
            out.append(ScoredNode(text=node.text, metadata=dict(node.metadata), score=score))
        return out


NO_CONTEXT_PROMPT = (
    "[INST] Answer briefly. "
    "If you do not know the answer, say so.\n\nQuestion: {question} [/INST]"
)


def answer_without_retrieval(backend, question: str) -> str:
    """The no-retrieval floor: same model, same decoding, zero context.

    Any case that passes here was answered from the model's weights, not from the
    documents — which is the check that separates retrieval from recall.
    """
    return backend.complete(NO_CONTEXT_PROMPT.format(question=question))
