import { formatDuration, formatSources } from '../format';
import type { QuestionProgress } from '../types';
import './QuestionList.css';

interface QuestionListProps {
  /** Question text, parsed from the uploaded file so it is never echoed back. */
  questions: string[];
  /** Keyed by 1-based position; absent means the row has not finished. */
  progress: Record<number, QuestionProgress>;
}

/** The questions as they land, so a long run shows work rather than a spinner. */
export function QuestionList({ questions, progress }: QuestionListProps) {
  const finished = Object.keys(progress).length;

  return (
    <section className="question-list" aria-label="Questions">
      <p className="question-list__caption">
        <span className="question-list__count">{finished} done</span>
        <span className="question-list__sep">·</span>
        {questions.length - finished} remaining
      </p>
      <ol className="question-list__rows">
        {questions.map((question, i) => {
          const row = progress[i + 1] ?? { status: 'queued' as const };
          return (
            <li key={i} className={`question-row question-row--${row.status}`}>
              <span className="question-row__index">{String(i + 1).padStart(2, '0')}</span>
              <span className="question-row__status">{row.status}</span>
              <span className="question-row__text">{question}</span>
              <span className="question-row__source">
                {row.status === 'done' ? formatSources(row.sources) : ''}
              </span>
              <span className="question-row__ms">
                {row.ms === undefined ? '' : formatDuration(row.ms)}
              </span>
            </li>
          );
        })}
      </ol>
    </section>
  );
}
