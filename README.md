# Document Q&A

[![tests](https://github.com/krisseo0202/zania_takehome/actions/workflows/tests.yml/badge.svg)](https://github.com/krisseo0202/zania_takehome/actions/workflows/tests.yml)

Answers a list of questions from one uploaded document, and only from that
document. A question the document does not cover is answered
`Data Not Available` rather than guessed.

## Status

Complete and tested: load → chunk → index → retrieve → answer, behind a
FastAPI service. Answer quality has been read by hand, not measured against a
labelled set.

```text
main.py          FastAPI app: POST /answer, GET /health
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

Nothing to install but the key. Copy `.env.example` to `.env`, paste your
OpenAI key into `OPENAI_API_KEY`, then:

```bash
docker compose up --build        # API on http://localhost:8000
```

Without Docker, or to run the tests and the CLI scripts, use a venv instead:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn main:app --port 8000
```

The sample document is not in the repo. Download it next to this README:

```bash
curl -O https://getnave.com/assets2/docs/Nave-SOC2-Type-2-Report.pdf
```

## Run

With the server up either way:

```bash
curl -X POST http://localhost:8000/answer \
  -F "questions_file=@samples/questions.json" \
  -F "document_file=@Nave-SOC2-Type-2-Report.pdf"
```

| Endpoint | Method | In | Out |
|---|---|---|---|
| `/answer` | POST, multipart | `questions_file` (JSON), `document_file` (PDF or JSON) | `200` question/answer pairs |
| `/answer/stream` | POST, multipart | same two files | `200` NDJSON, one stage per line, then the same body |
| `/health` | GET | — | status plus the settings in force |

Errors are explicit: `400` unreadable document or questions, `413` upload over
`MAX_FILE_MB`, `422` a missing form field, `502` OpenAI failed after its
bounded retries. A provider failure is never reported as `Data Not Available`.

Without the server, the same pipeline from the command line:

```bash
# Inspect what retrieval returns, before any chat model is involved.
.venv/bin/python scripts/run_sample.py --retrieval-only

# Full pipeline: questions in, question/answer pairs out.
.venv/bin/python scripts/run_sample.py Nave-SOC2-Type-2-Report.pdf samples/questions.json
```

Real output from `samples/sample_doc.json`, abridged:

```json
{"document": "sample_doc.json",
 "results": [
   {"question": "Which cloud provider hosts the service?",
    "answer": "Amazon Web Services (AWS) hosts the service.",
    "confidence": "high",
    "sources": ["json:hosting", "json:security", "json:incident_response"]},
   {"question": "What is the incident notification SLA in hours?",
    "answer": "Data Not Available",
    "confidence": "low",
    "sources": ["json:incident_response", "json:monitoring"]},
   {"question": "Which are performed: APM, EUM, and DEM?",
    "answer": "APM is performed as it is stated that \"Application Performance Monitoring is enabled.\" EUM is not performed as it is explicitly stated that EUM is false. There is no information provided regarding DEM.",
    "confidence": "medium",
    "sources": ["json:monitoring", "json:hosting"]}],
 "usage": {"input_tokens": 2800, "output_tokens": 106, "embedding_tokens": 92,
           "estimated_cost_usd": 0.000485}}
```

The last answer is the one to look at: the document says EUM is `false` and
says nothing at all about DEM, and the answer keeps those apart. Note it
phrases the missing part as prose rather than the literal `Data Not Available`.
Only a whole-answer abstention is guaranteed to be the exact string, which is
why callers should filter on `answer == "Data Not Available"` and treat
per-part markers as text.

`confidence` is the model's own `high` / `medium` / `low` rating of how directly
the excerpts state the answer; an abstention is always `low`. It is a
self-report, useful for sorting review effort, not a calibrated probability.
`sources` lists the passages **retrieved** for that question, best match first
(`page:14`, `json:hosting`). It is the evidence put in front of the model, not
a verified citation of what the answer used, and it is present even when the
answer abstains. `usage.estimated_cost_usd` applies the list prices in
`config.py` to real token counts: an estimate, not a bill.

## Inputs

**Questions** — `.json` only, checked by extension before the body is read. `["q1", "q2"]`, `{"questions": [...]}`, or
`[{"question": ...}]`. Order and duplicates are preserved; a duplicate is
answered once and repeated in the output. Anything else is rejected before a
token is spent.

**Document** — `.pdf` or `.json`, likewise checked by extension. A PDF becomes one section per page
(`page:12`); a JSON document becomes one section per top-level key, flattened to
`hosting.provider: "AWS"` lines (`json:hosting`). `false`, `0` and `null`
survive flattening — an explicit "no" is evidence, not an absence.

## How it works

Each section is split into ~500-character overlapping chunks that keep their
page or path, embedded once into a per-upload Chroma collection, and the five
nearest chunks per question are passed to `gpt-4o-mini` as the only permitted
evidence. Distinct questions are answered on up to five threads
(`max_concurrency`), duplicates cost nothing, and the response keeps input
order. The collection is deleted when the run ends, including on error.

## Design decisions

| Decision | Why |
|---|---|
| Index per run, discarded after | Two files in, pairs out. Persistence would add collection lifecycle and cleanup for no benefit here. |
| Chroma, in-process | No infrastructure to stand up. Swappable behind LangChain's vector store interface. |
| 500-char chunks, 150 overlap, k=5 | `top_k` fixes how many chunks reach the model, not how big they are, so 500 costs less per question than 1000 with no measured quality loss (`examples/`). Overlap and k are defaults to tune against an eval set, not truths. |
| One call per *distinct* question | Repeated questions are common in questionnaires and cost nothing to reuse. |
| `Data Not Available` as a literal | Deterministic, so callers can filter and count it, and tests can assert on it. |
| Provider errors raise | A timeout must not be reported as missing evidence. Abstention means retrieval found nothing. |

## Streaming progress

`/answer/stream` does the same work as `/answer` and reports it stage by stage,
one JSON object per line:

```text
{"stage": "questions", "count": 5}
{"stage": "load", "sections": 84}
{"stage": "chunk", "chunks": 751, "chunk_size": 500, "chunk_overlap": 150}
{"stage": "index", "vectors": 751, "top_k": 5}
{"stage": "generating", "positions": [1]}
{"stage": "answer", "done": 1, "total": 19, "positions": [1],
 "outcome": "abstain", "confidence": "low", "sources": ["page:81"], "ms": 2140}
{"stage": "done", "result": { ... same body as /answer ... }}
```

`generating` is the exception to the rule below: it fires when a question
reaches the chat model, so a question that finds no evidence is never shown as
generating. Every other stage is emitted **after** it finishes, so the progress bar reports work
that happened rather than work that is hoped for. Answering runs concurrently,
so `positions` names the rows an event completes rather than assuming order; a
duplicate question is answered once and marks every row it occupies. A failure arrives as
`{"stage": "error", "status": 400, ...}` instead of a partial result. NDJSON
rather than server-sent events, because `EventSource` is GET-only and cannot
carry the upload.

## Web front end (optional)

`web/` is a small React page over the same API. The Docker image builds it
and the service mounts `web/dist` at `/`; without a build, `/` redirects to
the API docs. To build it for the venv path:

```bash
cd web && npm install && npm run build   # needs Node >= 20.19 (see web/.nvmrc)
```

Node 18 fails with a `node:util` `styleText` error: Vite 8 requires Node 20.19+
and npm does not enforce `engines` by default, so the build dies mid-flight
rather than up front. `web/.npmrc` sets `engine-strict=true` to make it refuse
early instead.

`/health` publishes `fallback` and `max_file_mb` so the page counts abstentions
and pre-checks upload size against the running server's settings rather than a
second hardcoded copy.

## Tests

```bash
.venv/bin/python -m pytest -q      # 73 tests, no API key, no network (2 skip without the Nave PDF)
```

Fake embeddings and stub models test the pipeline's contracts — ordering,
deduplication, cleanup, abstention, provenance. They do not test retrieval
quality or whether the real model abstains correctly; that needs the live run
above and a labelled set.

## Observed behaviour on the sample report

On the 84-page Nave SOC 2 report with the 19 sample questions: 333 chunks
indexed, 19 answers in 17 s, 31.5k input + 578 output + 78k embedding tokens,
about $0.007. **7 answered, 12 abstained.**

All seven answers are "Partially", at medium confidence, and that is the honest
shape of the task: a SOC 2 report describes audited controls, not questionnaire
answers, so it usually supports a broader claim than the question asks about.
Asked whether *asset management and malware protection* policies are reviewed
annually, the report says company policies are reviewed annually - related, not
the same claim. The prompt names the gap instead of rounding it up to "Yes".

The abstentions were checked by hand against the document, not assumed:

- Retrieval returns the right pages (business continuity → p82, incident
  response → p77-79, access control → p59-63), so these are not search misses.
- "Where are your data centres located?" abstains correctly. The report names
  the provider - "Nave's application runs in the Google Cloud Platform (GCP)",
  p16 - but never states a location or region, and the question asks where.

`sample_json.csv` is **not** ground truth for this PDF: it answers a different
document (it cites `Company_kb (1).json`), so its answers are not achievable
from this report. No labelled set for this document exists yet, which is why
every quality claim above is a hand check rather than a measurement.

## Chunk-size comparison

`examples/` holds ten runs of the same 19 questions - `chunk_size` 500/1000/1500,
`chunk_overlap` 10/20/30%, and `top_k` 5/6/7/8 - with `examples/README.md`
reading them. Short version: chunk size and `top_k` move cost (smaller and
fewer are cheaper; cost is linear in k), overlap barely moves it, and none of
them moves the answered/abstained split by more than the run-to-run variance.
`chunk_size=500, top_k=5` is the default on the cost evidence, not on quality.

## Limitations

- No OCR: a scanned PDF with no text layer is rejected, not read.
- Answer quality has not been measured against a labelled set.
- No defence against instructions embedded in an uploaded document. The prompt
  tells the model to treat excerpts as evidence, which is not a guarantee.
- Dense retrieval only, so a question phrased unlike the document may miss.
- Facts spread across many pages, and tables split across chunks, retrieve poorly.
- Citations (`[page:12]`) are written by the model, not verified against the
  excerpts; `sources` is the checked list of what it was shown.
