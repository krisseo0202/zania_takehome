import { useState } from 'react';
import './FileField.css';

interface FileFieldProps {
  id: string;
  accept: string;
  file: File | null;
  metaText: string | null;
  disabled: boolean;
  onSelect: (file: File | null) => void;
}

/** One `questions_file` / `document_file` picker, styled to match the mockup's
 * run bar. Used twice with different ids/accept types in RunForm. */
export function FileField({ id, accept, file, metaText, disabled, onSelect }: FileFieldProps) {
  const [isDragging, setIsDragging] = useState(false);

  function handleDrop(event: React.DragEvent) {
    event.preventDefault();
    setIsDragging(false);
    if (disabled) return;
    const dropped = event.dataTransfer.files?.[0];
    if (dropped) onSelect(dropped);
  }

  return (
    <div
      className={`file-field${isDragging ? ' file-field--dragging' : ''}`}
      onDragOver={(event) => {
        // Without preventDefault the browser navigates to the dropped file.
        event.preventDefault();
        if (!disabled) setIsDragging(true);
      }}
      onDragLeave={(event) => {
        // Ignore drags moving between children of this field.
        if (!event.currentTarget.contains(event.relatedTarget as Node)) setIsDragging(false);
      }}
      onDrop={handleDrop}
    >
      <span className="file-field__label">{id}</span>
      <div className="file-field__row">
        {file ? (
          <>
            <span className="file-field__name">{file.name}</span>
            {metaText && <span className="file-field__meta">{metaText}</span>}
          </>
        ) : (
          <span className="file-field__placeholder">No file selected</span>
        )}
        <label
          htmlFor={id}
          className={`file-field__replace${disabled ? ' file-field__replace--disabled' : ''}`}
        >
          {file ? 'Replace' : 'Choose file'}
        </label>
        <span className="file-field__hint" aria-hidden="true">or drop it here</span>
        <input
          id={id}
          name={id}
          type="file"
          accept={accept}
          className="file-field__input"
          disabled={disabled}
          onChange={(event) => onSelect(event.target.files?.[0] ?? null)}
        />
      </div>
    </div>
  );
}
