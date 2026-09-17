# Document Q&A

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
.venv/bin/uvicorn main:app --port 8000

curl -X POST http://localhost:8000/answer \
  -F "questions_file=@samples/questions.json" \
  -F "document_file=@Nave-SOC2-Type-2-Report.pdf"
```

| Endpoint | Method | In | Out |
|---|---|---|---|
| `/answer` | POST, multipart | `questions_file` (JSON), `document_file` (PDF or JSON) | `200` question/answer pairs |
| `/health` | GET | — | `{"status": "ok"}` |

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

Output shape:

Real output from `samples/sample_doc.json`, abridged:

```json
{"document": "sample_doc.json",
 "results": [
   {"question": "Which cloud provider hosts the service?",
    "answer": "Amazon Web Services (AWS)",
    "confidence": "high",
    "sources": ["json:hosting", "json:security", "json:incident_response"]},
   {"question": "What is the incident notification SLA in hours?",
    "answer": "Data Not Available",
    "confidence": "low",
    "sources": ["json:incident_response", "json:monitoring"]}],
 "usage": {"input_tokens": 1360, "output_tokens": 55, "embedding_tokens": 92,
           "estimated_cost_usd": 0.000239}}
```

`confidence` is the model's own `high` / `medium` / `low` rating of how directly
the excerpts state the answer; an abstention is always `low`. It is a
self-report, useful for sorting review effort, not a calibrated probability.
`sources` lists the passages **retrieved** for that question, best match first
(`page:14`, `json:hosting`). It is the evidence put in front of the model, not
a verified citation of what the answer used, and it is present even when the
answer abstains. `usage.estimated_cost_usd` applies the list prices in
`config.py` to real token counts: an estimate, not a bill.

The third answer is the one to look at: the document says EUM is `false` and
says nothing at all about DEM, and those are different answers.

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

## Web front end (optional)

`web/` is a small React page over the same API; the service mounts `web/dist`
at `/` when that build exists, so the API alone needs none of it.

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
.venv/bin/python -m pytest -q      # 57 tests, no API key, no network
```

Fake embeddings and stub models test the pipeline's contracts — ordering,
deduplication, cleanup, abstention, provenance. They do not test retrieval
quality or whether the real model abstains correctly; that needs the live run
above and a labelled set.

## Observed behaviour on the sample report

On the 84-page Nave SOC 2 report with the 19 sample questions: 333 chunks
indexed in 21 s, 19 answers in 15 s, about a cent. **18 of 19 abstained.**

That is mostly correct, not a failure to find evidence. Retrieval was checked
by hand and returned the right pages (business continuity → p82, incident
response → p77-79, access control → p59-63), but a SOC 2 report describes
audited controls, not questionnaire answers: it says a Business Continuity Plan
exists and is tested, never that it is "available to authorized stakeholders".
The prompt's "do not treat 'not stated' as 'no'" rule abstains on the gap.

The one grounded answer (unauthorized software) traces to control CC6.8 on
pages 68-69. Note that `sample_json.csv` answers a *different* document — it
cites GCP and `Company_kb (1).json` — so it is not ground truth for this PDF,
and no labelled set for this document exists yet.

## Limitations

- No OCR: a scanned PDF with no text layer is rejected, not read.
- Answer quality has not been measured against a labelled set.
- No defence against instructions embedded in an uploaded document. The prompt
  tells the model to treat excerpts as evidence, which is not a guarantee.
- Dense retrieval only, so a question phrased unlike the document may miss.
- Facts spread across many pages, and tables split across chunks, retrieve poorly.
- No citations in the answer; page and path IDs stay in the logs.
