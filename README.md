# Document Q&A

Answers a list of questions from one uploaded document, and only from that
document. A question the document does not cover is answered
`Data Not Available` rather than guessed.

## Status

The pipeline is complete and tested: load → chunk → index → retrieve → answer.
The HTTP API is not built yet, so the entry point today is a CLI script.
`DocumentQAService.answer_document()` is the single call a route will wrap.

```text
app/
  config.py      settings from .env, OpenAI clients with timeouts and retries
  loaders.py     uploads -> Documents: parse questions, read PDF/JSON, chunk
  retrieval.py   per-upload Chroma collection, similarity search, cleanup
  answering.py   prompt, one grounded answer, all answers in input order
  service.py     one upload end to end
tests/           one file per module, plus shared fakes in conftest.py
scripts/         run_sample.py, the CLI demo
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env          # paste your OpenAI key into OPENAI_API_KEY
```

The sample document is not in the repo. Download it next to this README:

```bash
curl -O https://getnave.com/assets2/docs/Nave-SOC2-Type-2-Report.pdf
```

## Run

```bash
# Inspect what retrieval returns, before any chat model is involved.
.venv/bin/python scripts/run_sample.py --retrieval-only

# Full pipeline: questions in, question/answer pairs out.
.venv/bin/python scripts/run_sample.py Nave-SOC2-Type-2-Report.pdf samples/questions.json
```

Output shape:

```json
{"document": "toy.json",
 "results": [{"question": "Which cloud provider hosts the service?", "answer": "AWS"}]}
```

## Inputs

**Questions** — JSON only. `["q1", "q2"]`, `{"questions": [...]}`, or
`[{"question": ...}]`. Order and duplicates are preserved; a duplicate is
answered once and repeated in the output. Anything else is rejected before a
token is spent.

**Document** — `.pdf` or `.json`. A PDF becomes one section per page
(`page:12`); a JSON document becomes one section per top-level key, flattened to
`hosting.provider: "AWS"` lines (`json:hosting`). `false`, `0` and `null`
survive flattening — an explicit "no" is evidence, not an absence.

## How it works

Each section is split into ~1000-character overlapping chunks that keep their
page or path, embedded once into a per-upload Chroma collection, and the five
nearest chunks per question are passed to `gpt-4o-mini` as the only permitted
evidence. The collection is deleted when the run ends, including on error.

## Design decisions

| Decision | Why |
|---|---|
| Index per run, discarded after | Two files in, pairs out. Persistence would add collection lifecycle and cleanup for no benefit here. |
| Chroma, in-process | No infrastructure to stand up. Swappable behind LangChain's vector store interface. |
| 1000-char chunks, 150 overlap, k=5 | A questionnaire answer is usually a paragraph or two. Defaults to tune against an eval set, not truths. |
| One call per *distinct* question | Repeated questions are common in questionnaires and cost nothing to reuse. |
| `Data Not Available` as a literal | Deterministic, so callers can filter and count it, and tests can assert on it. |
| Provider errors raise | A timeout must not be reported as missing evidence. Abstention means retrieval found nothing. |

## Tests

```bash
.venv/bin/python -m pytest -q      # 40 tests, no API key, no network
```

Fake embeddings and stub models test the pipeline's contracts — ordering,
deduplication, cleanup, abstention, provenance. They do not test retrieval
quality or whether the real model abstains correctly; that needs the live run
above and a labelled set.

## Limitations

- No OCR: a scanned PDF with no text layer is rejected, not read.
- Answer quality has not been measured against a labelled set.
- No defence against instructions embedded in an uploaded document. The prompt
  tells the model to treat excerpts as evidence, which is not a guarantee.
- Dense retrieval only, so a question phrased unlike the document may miss.
- Facts spread across many pages, and tables split across chunks, retrieve poorly.
- No citations in the answer; page and path IDs stay in the logs.
