"""Run the pipeline end to end on two local files.

    python scripts/run_sample.py [document] [questions.json] [--retrieval-only]

Needs OPENAI_API_KEY (in .env). `--retrieval-only` prints the retrieved source
IDs and text without calling the chat model: if the expected evidence is not
retrieved, fix extraction, chunking or search here -- not the answer prompt.
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import get_chat_model, get_embeddings
from rag import DocumentQAService, chunk_documents, load_document, open_index, parse_questions, retrieve

PREVIEW_CHARS = 200


def main(doc_path: str, questions_path: str, retrieval_only: bool = False) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")

    document = Path(doc_path)
    embeddings = get_embeddings()
    if retrieval_only:
        questions = parse_questions(Path(questions_path).read_bytes())
        chunks = chunk_documents(load_document(document.name, document.read_bytes()))
        with open_index(chunks, embeddings) as store:
            for question in questions:
                print(f"\n=== {question}")
                for passage in retrieve(store, question):
                    text = " ".join(passage.page_content.split())
                    print(f"  [{passage.metadata['source_id']}] {text[:PREVIEW_CHARS]}")
        return

    llm = get_chat_model()
    response = DocumentQAService(embeddings, llm).answer_document(
        document.name, document.read_bytes(), Path(questions_path).read_bytes()
    )
    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    doc, qs = args or ["samples/sample_doc.json", "samples/toy_questions.json"]
    main(doc, qs, retrieval_only="--retrieval-only" in sys.argv)
