import type { QuestionProgress, StageEvent } from '../types';
import './StageBar.css';

export interface StageProgress {
  questions?: number;
  sections?: number;
  chunks?: number;
  chunkSize?: number;
  chunkOverlap?: number;
  vectors?: number;
  topK?: number;
  answered?: number;
  total?: number;
  /** Per-row outcome, keyed by 1-based position. */
  rows?: Record<number, QuestionProgress>;
}

/** Apply one row state to every position an event names. */
function markRows(
  current: StageProgress,
  event: StageEvent,
  row: QuestionProgress,
): Record<number, QuestionProgress> {
  const rows = { ...(current.rows ?? {}) };
  for (const position of event.positions ?? []) rows[position] = row;
  return rows;
}

/** Fold a stage event into the progress state. */
export function applyStage(current: StageProgress, event: StageEvent): StageProgress {
  switch (event.stage) {
    case 'questions':
      return { ...current, questions: event.count, total: event.count };
    case 'load':
      return { ...current, sections: event.sections };
    case 'chunk':
      return {
        ...current,
        chunks: event.chunks,
        chunkSize: event.chunk_size,
        chunkOverlap: event.chunk_overlap,
      };
    case 'index':
      return { ...current, vectors: event.vectors, topK: event.top_k };
    case 'generating':
      return { ...current, rows: markRows(current, event, { status: 'generating' }) };
    case 'answer':
      return {
        ...current,
        answered: event.done,
        total: event.total ?? current.total,
        rows: markRows(current, event, {
          status: event.outcome ?? 'done',
          confidence: event.confidence,
          sources: event.sources,
          ms: event.ms,
        }),
      };
    default:
      return current;
  }
}

interface Stage {
  name: string;
  /** Detail line, or null while the stage has not reported yet. */
  detail: string | null;
  /** 0 to 1. Only the answering stage is partially complete. */
  fraction: number;
}

function stages(p: StageProgress): Stage[] {
  const answered = p.answered ?? 0;
  const total = p.total ?? 0;
  return [
    {
      name: 'Load',
      detail: p.sections === undefined ? null : `${p.sections} sections`,
      fraction: p.sections === undefined ? 0 : 1,
    },
    {
      name: 'Chunk',
      detail: p.chunks === undefined ? null : `${p.chunks} chunks`,
      fraction: p.chunks === undefined ? 0 : 1,
    },
    {
      name: 'Index',
      detail: p.vectors === undefined ? null : `${p.vectors} vectors · k=${p.topK}`,
      fraction: p.vectors === undefined ? 0 : 1,
    },
    {
      name: 'Answer',
      detail: total ? `${answered} of ${total}` : null,
      fraction: total ? answered / total : 0,
    },
  ];
}

/** Real progress: every value comes from a stage the server finished. */
export function StageBar({ progress }: { progress: StageProgress }) {
  const list = stages(progress);
  const done = list.filter((s) => s.fraction >= 1).length;

  return (
    <section
      className="stage-bar"
      aria-label="Pipeline progress"
      role="progressbar"
      aria-valuemin={0}
      aria-valuemax={list.length}
      aria-valuenow={done}
      aria-valuetext={`${done} of ${list.length} stages complete`}
    >
      {list.map((stage) => (
        <div
          key={stage.name}
          className={`stage${stage.fraction >= 1 ? ' stage--done' : stage.fraction > 0 ? ' stage--active' : ''}`}
        >
          <span className="stage__name">{stage.name}</span>
          <span className="stage__detail">{stage.detail ?? '—'}</span>
          <span className="stage__track">
            <span className="stage__fill" style={{ width: `${Math.round(stage.fraction * 100)}%` }} />
          </span>
        </div>
      ))}
    </section>
  );
}
