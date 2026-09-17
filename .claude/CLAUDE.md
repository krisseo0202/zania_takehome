# Project instructions

- Build a document Q&A API accepting one PDF/JSON document
  and a JSON list of questions.
- Use gpt-4o-mini for answer generation.
- Answer only from retrieved document evidence.
- Use "Data Not Available" when evidence is insufficient.
- Preserve question order and duplicates in the response.
- Keep document indexes isolated per request and clean them up.
- Never commit credentials or log document contents.
- Keep automated tests offline; use fake model dependencies.
- Avoid adding features beyond the take-home requirements.