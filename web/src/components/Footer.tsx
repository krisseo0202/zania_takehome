import './Footer.css';

interface FooterProps {
  hasResult: boolean;
  onDownloadJson: () => void;
  onDownloadCsv: () => void;
}

export function Footer({ hasResult, onDownloadJson, onDownloadCsv }: FooterProps) {
  return (
    <footer className="app-footer">
      <span className="app-footer__note">
        Every answer is drawn only from the uploaded document. Nothing is carried over between runs.
      </span>
      <span className="app-footer__spacer" />
      <button type="button" className="app-footer__button" disabled={!hasResult} onClick={onDownloadJson}>
        Download JSON
      </button>
      <button type="button" className="app-footer__button" disabled={!hasResult} onClick={onDownloadCsv}>
        Download CSV
      </button>
    </footer>
  );
}
