import { formatDuration, formatUsd } from '../format';
import './SummaryStrip.css';

interface SummaryStripProps {
  total: number;
  answered: number;
  missing: number;
  wallClockMs: number;
  estimatedCostUsd?: number;
}

export function SummaryStrip({
  total,
  answered,
  missing,
  wallClockMs,
  estimatedCostUsd,
}: SummaryStripProps) {
  return (
    <section className="summary-strip" aria-label="Run summary">
      <div className="summary-strip__stat">
        <span className="summary-strip__value">{total}</span>
        <span className="summary-strip__label">questions</span>
      </div>
      <div className="summary-strip__stat summary-strip__stat--bordered">
        <span className="summary-strip__value">{answered}</span>
        <span className="summary-strip__label">answered</span>
      </div>
      <div className="summary-strip__stat summary-strip__stat--bordered">
        <span className="summary-strip__value summary-strip__value--amber">{missing}</span>
        <span className="summary-strip__label">Data Not Available</span>
      </div>
      <div className="summary-strip__stat summary-strip__stat--bordered">
        <span className="summary-strip__value summary-strip__value--mono">{formatDuration(wallClockMs)}</span>
        <span className="summary-strip__label">wall clock</span>
      </div>
      {estimatedCostUsd !== undefined && (
        <div className="summary-strip__stat summary-strip__stat--bordered">
          <span className="summary-strip__value summary-strip__value--mono">
            {formatUsd(estimatedCostUsd)}
          </span>
          {/* List prices, so an estimate rather than a bill. */}
          <span className="summary-strip__label">est. spend</span>
        </div>
      )}
    </section>
  );
}
