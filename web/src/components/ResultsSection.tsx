import type { RefObject } from 'react';
import type { AnswerResponse } from '../types';
import type { Phase } from '../App';
import { ErrorPanel, type ErrorInfo } from './ErrorPanel';
import { FilterChips, type FilterId } from './FilterChips';
import { StageBar, type StageProgress } from './StageBar';
import { ResultsTable } from './ResultsTable';
import { SummaryStrip } from './SummaryStrip';
import './ResultsSection.css';

interface ResultsSectionProps {
  phase: Phase;
  error: ErrorInfo | null;
  result: AnswerResponse | null;
  elapsedMs: number | null;
  progress: StageProgress;
  /** The exact abstention literal, as published by /health. */
  fallback: string;
  filter: FilterId;
  onFilterChange: (filter: FilterId) => void;
  headingRef: RefObject<HTMLHeadingElement | null>;
}

/** Everything below the run bar: empty in the idle state, otherwise whichever
 * of running / error / success applies. This is the single aria-live region
 * that announces a run's outcome, and it holds the heading focus moves to
 * when a run finishes. */
export function ResultsSection({
  phase,
  error,
  result,
  elapsedMs,
  progress,
  fallback,
  filter,
  onFilterChange,
  headingRef,
}: ResultsSectionProps) {
  if (phase === 'idle') return null;

  if (phase === 'running') {
    return (
      <section className="results-section" aria-live="polite">
        <h2 className="visually-hidden" tabIndex={-1} ref={headingRef}>
          Running
        </h2>
        <StageBar progress={progress} />
      </section>
    );
  }

  if (phase === 'error' && error) {
    return (
      <section className="results-section" aria-live="polite">
        <h2 className="results-section__error-heading" tabIndex={-1} ref={headingRef}>
          Request failed
        </h2>
        <ErrorPanel error={error} />
      </section>
    );
  }

  if (phase === 'success' && result && elapsedMs !== null) {
    const rows = result.results.map((item, i) => ({
      index: i + 1,
      question: item.question,
      answer: item.answer,
      isMissing: item.answer === fallback,
      sources: item.sources,
    }));
    const answered = rows.filter((row) => !row.isMissing).length;
    const missing = rows.length - answered;
    const counts: Record<FilterId, number> = { all: rows.length, answered, missing };
    const shown = rows.filter((row) => {
      if (filter === 'answered') return !row.isMissing;
      if (filter === 'missing') return row.isMissing;
      return true;
    });

    return (
      <section className="results-section" aria-live="polite">
        <h2 className="visually-hidden" tabIndex={-1} ref={headingRef}>
          Results
        </h2>
        <SummaryStrip
          total={rows.length}
          answered={answered}
          missing={missing}
          wallClockMs={elapsedMs}
          estimatedCostUsd={result.usage?.estimated_cost_usd}
        />
        <FilterChips active={filter} counts={counts} onChange={onFilterChange} />
        <ResultsTable rows={shown} />
      </section>
    );
  }

  return null;
}
