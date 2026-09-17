import './ErrorPanel.css';

export interface ErrorInfo {
  status: number | null;
  detail: string;
}

/** No role="alert" here on purpose: the parent ResultsSection is already an
 * aria-live="polite" region, and "alert" would make it assertive instead. */
export function ErrorPanel({ error }: { error: ErrorInfo }) {
  return (
    <div className="error-panel">
      {error.status !== null && <span className="error-panel__status">{error.status}</span>}
      <span className="error-panel__detail">{error.detail}</span>
    </div>
  );
}
