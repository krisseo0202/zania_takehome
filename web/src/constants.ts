// The exact literal `answer_document()` returns when no evidence supports an
// answer (see `settings.fallback` in config.py / FALLBACK in rag.py). The API
// response doesn't carry this value itself, so the front end hardcodes the
// same string the backend is contractually fixed to.
export const DATA_NOT_AVAILABLE = 'Data Not Available';

// Matches the API's 413 file-size limit, so we can reject an oversized file
// before spending a network round trip on it.
export const MAX_FILE_BYTES = 20 * 1024 * 1024;
