// Mirrors the Pydantic response models the FastAPI route returns.

export interface AnswerResultItem {
  question: string;
  answer: string;
  /** Passages retrieved for this question, best match first. Evidence that was
   * put in front of the model, not a verified citation of what it used. */
  sources?: string[];
}

export interface RunUsage {
  input_tokens: number;
  output_tokens: number;
  embedding_tokens: number;
  estimated_cost_usd: number;
}

export interface AnswerResponse {
  document: string;
  results: AnswerResultItem[];
  usage?: RunUsage;
}

// Our handlers send a string; FastAPI's own 422 sends a list of field errors,
// so the reader must check the type rather than trust it.
export interface ApiErrorBody {
  detail: string | unknown[];
}

export interface ServerConfig {
  /** The exact literal the API uses for an unsupported answer. */
  fallback: string;
  maxFileBytes: number;
  /** "gpt-4o-mini · text-embedding-3-small · k=5 · temp 0", from the server. */
  modelLine: string;
}


/** One line of /answer/stream. Every stage is emitted after it completes. */
export interface StageEvent {
  stage: 'questions' | 'load' | 'chunk' | 'index' | 'answer' | 'done' | 'error';
  count?: number;
  sections?: number;
  chunks?: number;
  chunk_size?: number;
  chunk_overlap?: number;
  vectors?: number;
  top_k?: number;
  done?: number;
  total?: number;
  result?: AnswerResponse;
  status?: number;
  detail?: string;
}
