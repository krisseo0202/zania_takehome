import './ProgressBar.css';

/** Indeterminate progress affordance shown while a run is in flight. */
export function ProgressBar() {
  return (
    <div className="progress-bar" role="progressbar" aria-label="Answering questions" aria-busy="true">
      <div className="progress-bar__fill" />
    </div>
  );
}
