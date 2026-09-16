"""RAG pipeline for the Plumber case-file assignment.

Usage:
    python final/rag_pipeline.py
    python final/rag_pipeline.py --query "..."

The script performs local PDF retrieval and uses an OpenAI-compatible chat API
only for extracting the requested code from the retrieved evidence.
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from openai import OpenAI
from pypdf import PdfReader


DEFAULT_KB = Path(__file__).parent / "drive-download-20260916T124022Z-1-001" / "knowledge base data"

QUERIES = [
    "According to incident PCF-0091 at Plumber Orbital Holding Ring 4, what was the Containment Override Code issued for Vilgax during the breach?",
    "In the Bellwood Regional Veterinary Annex incident PCF-0114 involving Dr. Animo, what Bio-Hybrid Diagnostic Code was recorded?",
    "For the Undertown Sub-Level Vault 7 theft filed under PCF-0158 involving the Forever Knights Relic Chapter, what Relic Retrieval Code was assigned?",
    "During the Null Void Relay Station 2 malfunction logged as PCF-0173, what Portal Stabilization Code was applied to the containment grid?",
    "In the Galvan Prime Remote Diagnostics Lab report PCF-0206 on Omnitrix DNA Archive Core corruption, what DNA Archive Decryption Code was issued?",
]


@dataclass(frozen=True)
class Document:
    source: str
    text: str


def load_documents(kb_path: Path) -> list[Document]:
    """Extract text from every PDF in the knowledge base."""
    documents = []
    for pdf_path in sorted(kb_path.glob("*.pdf")):
        reader = PdfReader(str(pdf_path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        if text.strip():
            documents.append(Document(pdf_path.name, text))
    if not documents:
        raise FileNotFoundError(f"No readable PDF files found in {kb_path}")
    return documents


def terms(value: str) -> set[str]:
    return {term for term in re.findall(r"[a-z0-9]+", value.lower()) if len(term) > 2}


def retrieve(query: str, documents: Iterable[Document], top_k: int = 3) -> list[Document]:
    """Rank documents by query-term overlap, with incident IDs weighted heavily."""
    query_terms = terms(query)
    incident_ids = set(re.findall(r"PCF-\d+", query.upper()))

    scored = []
    for document in documents:
        document_terms = terms(document.text)
        score = len(query_terms & document_terms)
        score += 20 * sum(incident_id in document.text.upper() for incident_id in incident_ids)
        scored.append((score, document))

    return [document for _, document in sorted(scored, key=lambda item: item[0], reverse=True)[:top_k]]


def extract_code(client: OpenAI, query: str, evidence: list[Document], model: str) -> str:
    context = "\n\n".join(f"SOURCE: {doc.source}\n{doc.text}" for doc in evidence)
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract the single code requested by the user from the supplied case files. "
                    "Return only the code fragment, with no explanation, punctuation, or markdown. "
                    "Do not infer or invent a code."
                ),
            },
            {"role": "user", "content": f"Question:\n{query}\n\nCase files:\n{context}"},
        ],
    )
    code = response.choices[0].message.content or ""
    code = code.strip().strip("` ").splitlines()[0].strip()
    if not re.fullmatch(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*", code):
        raise ValueError(f"Model returned an invalid code fragment: {code!r}")
    return code


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Plumber case-file RAG pipeline.")
    parser.add_argument("--query", help="Run one custom query instead of all five assignment queries.")
    parser.add_argument("--kb", type=Path, default=DEFAULT_KB, help="Directory containing the source PDFs.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of retrieved PDFs sent to the model.")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY before running the pipeline.")

    documents = load_documents(args.kb)
    base_url = os.getenv("OPENAI_BASE_URL")
    if not base_url and api_key.startswith("sk-hc-"):
        base_url = "https://ai.hackclub.com/proxy/v1"
    client_options = {"api_key": api_key}
    if base_url:
        client_options["base_url"] = base_url
    client = OpenAI(**client_options)
    queries = [args.query] if args.query else QUERIES
    codes = []

    for query in queries:
        evidence = retrieve(query, documents, top_k=args.top_k)
        code = extract_code(client, query, evidence, args.model)
        print(f"{query}\n{code}\n")
        codes.append(code)

    if not args.query:
        print(f"SUBMISSION_CODE={'-'.join(codes)}")


if __name__ == "__main__":
    main()