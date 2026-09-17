import { FileField } from './FileField';
import './RunForm.css';

interface RunFormProps {
  questionsFile: File | null;
  documentFile: File | null;
  questionsMeta: string | null;
  documentMeta: string | null;
  isRunning: boolean;
  submitLabel: string;
  onQuestionsFile: (file: File | null) => void;
  onDocumentFile: (file: File | null) => void;
  onSubmit: () => void;
}

/** The two file pickers plus the submit button. A real <form> so file inputs,
 * labels, and the submit button behave the way assistive tech expects. */
export function RunForm({
  questionsFile,
  documentFile,
  questionsMeta,
  documentMeta,
  isRunning,
  submitLabel,
  onQuestionsFile,
  onDocumentFile,
  onSubmit,
}: RunFormProps) {
  const missing: string[] = [];
  if (!questionsFile) missing.push('a questions file');
  if (!documentFile) missing.push('a document file');
  const canSubmit = missing.length === 0 && !isRunning;

  return (
    <form
      className="run-form"
      onSubmit={(event) => {
        event.preventDefault();
        onSubmit();
      }}
    >
      <FileField
        id="questions_file"
        accept=".json,application/json"
        accepted={['.json']}
        file={questionsFile}
        metaText={questionsMeta}
        disabled={isRunning}
        onSelect={onQuestionsFile}
      />
      <FileField
        id="document_file"
        accept=".pdf,.json,application/pdf,application/json"
        accepted={['.pdf', '.json']}
        file={documentFile}
        metaText={documentMeta}
        disabled={isRunning}
        onSelect={onDocumentFile}
      />
      <div className="run-form__action">
        <button type="submit" className="run-form__submit" disabled={!canSubmit} aria-busy={isRunning}>
          {isRunning ? 'Running…' : submitLabel}
        </button>
        {missing.length > 0 && !isRunning && (
          <p className="run-form__hint">Choose {missing.join(' and ')} to run.</p>
        )}
      </div>
    </form>
  );
}
