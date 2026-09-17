import { DATA_NOT_AVAILABLE, MAX_FILE_BYTES } from './constants';
import type { AnswerResponse, ApiErrorBody, ServerConfig, StageEvent } from './types';

/** An error response from the API itself: {"detail": "..."} with a status code. */
export class ApiRequestError extends Error {
  readonly status: number;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = 'ApiRequestError';
    this.status = status;
  }
}

export const DEFAULT_CONFIG: ServerConfig = {
  fallback: DATA_NOT_AVAILABLE,
  maxFileBytes: MAX_FILE_BYTES,
  modelLine: '',
};

interface HealthBody {
  fallback?: unknown;
  max_file_mb?: unknown;
  chat_model?: unknown;
  embedding_model?: unknown;
  top_k?: unknown;
  temperature?: unknown;
}

/** The header states which models answered; read it from the server so it
 * cannot drift from the settings actually in force. */
function modelLineFrom(body: HealthBody): string {
  const parts = [body.chat_model, body.embedding_model].filter(
    (value): value is string => typeof value === 'string' && value.length > 0,
  );
  if (typeof body.top_k === 'number') parts.push(`k=${body.top_k}`);
  if (typeof body.temperature === 'number') parts.push(`temp ${body.temperature}`);
  return parts.join(' · ');
}

/** /health publishes the values the UI would otherwise hardcode. Falls back to
 * the compiled-in defaults if the endpoint is unreachable or malformed. */
export async function fetchServerConfig(): Promise<ServerConfig> {
  try {
    const response = await fetch('/health');
    if (!response.ok) return DEFAULT_CONFIG;
    const body = (await response.json()) as HealthBody;
    return {
      modelLine: modelLineFrom(body) || DEFAULT_CONFIG.modelLine,
      fallback: typeof body.fallback === 'string' && body.fallback
        ? body.fallback : DEFAULT_CONFIG.fallback,
      maxFileBytes: typeof body.max_file_mb === 'number' && body.max_file_mb > 0
        ? body.max_file_mb * 1024 * 1024 : DEFAULT_CONFIG.maxFileBytes,
    };
  } catch {
    return DEFAULT_CONFIG;
  }
}

export async function submitAnswer(questionsFile: File, documentFile: File): Promise<AnswerResponse> {
  const body = new FormData();
  body.append('questions_file', questionsFile);
  body.append('document_file', documentFile);

  const response = await fetch('/answer', { method: 'POST', body });

  if (!response.ok) {
    throw new ApiRequestError(response.status, await readErrorDetail(response));
  }
  return (await response.json()) as AnswerResponse;
}

async function readErrorDetail(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as ApiErrorBody;
    if (typeof body.detail === 'string' && body.detail) return body.detail;
  } catch {
    // Response body wasn't JSON (e.g. a proxy error page); fall through.
  }
  return `Request failed with status ${response.status}`;
}


/** POST the two files and report each pipeline stage as the server finishes it.
 *
 * NDJSON rather than EventSource: EventSource is GET-only and cannot carry an
 * upload. Returns the final response; throws ApiRequestError on an error event.
 */
export async function submitAnswerStreaming(
  questionsFile: File,
  documentFile: File,
  onStage: (event: StageEvent) => void,
  signal?: AbortSignal,
): Promise<AnswerResponse> {
  const body = new FormData();
  body.append('questions_file', questionsFile);
  body.append('document_file', documentFile);

  const response = await fetch('/answer/stream', { method: 'POST', body, signal });
  if (!response.ok || !response.body) {
    throw new ApiRequestError(response.status, await readErrorDetail(response));
  }

  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = '';
  let result: AnswerResponse | null = null;

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += value;

    // Keep the trailing partial line: a chunk can split a JSON object.
    const lines = buffer.split('\n');
    buffer = lines.pop() ?? '';
    for (const line of lines) {
      if (!line.trim()) continue;
      const event = JSON.parse(line) as StageEvent;
      if (event.stage === 'error') {
        throw new ApiRequestError(event.status ?? 500, event.detail ?? 'Request failed');
      }
      if (event.stage === 'done') result = event.result ?? null;
      else onStage(event);
    }
  }

  if (!result) throw new ApiRequestError(502, 'Stream ended before the answers arrived');
  return result;
}
