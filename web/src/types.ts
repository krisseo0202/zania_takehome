// Mirrors the Pydantic response models the FastAPI route returns.

export interface AnswerResultItem {
  question: string;
  answer: string;
  /** The model's own high/medium/low rating. Self-reported, not calibrated. */
  confidence?: string;
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
  stage: 'questions' | 'load' | 'chunk' | 'index' | 'generating' | 'answer' | 'done' | 'error';
  count?: number;
  sections?: number;
  chunks?: number;
  chunk_size?: number;
  chunk_overlap?: number;
  vectors?: number;
  top_k?: number;
  done?: number;
  total?: number;
  /** 1-based rows this event completes; a duplicate question marks several. */
  positions?: number[];
  outcome?: 'done' | 'abstain';
  confidence?: string;
  sources?: string[];
  ms?: number;
  result?: AnswerResponse;
  /** HTTP status, on an error event only. */
  status?: number;
  detail?: string;
}

/** What the live list knows about one question while the run is in flight. */
export interface QuestionProgress {
  status: 'queued' | 'generating' | 'done' | 'abstain';
  confidence?: string;
  sources?: string[];
  ms?: number;
}
