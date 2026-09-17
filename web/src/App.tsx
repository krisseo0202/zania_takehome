import { useEffect, useRef, useState } from 'react';
import { ApiRequestError, DEFAULT_CONFIG, fetchServerConfig, submitAnswer } from './api';
import './App.css';
import { Footer } from './components/Footer';
import { Header } from './components/Header';
import type { ErrorInfo } from './components/ErrorPanel';
import type { FilterId } from './components/FilterChips';
import { ResultsSection } from './components/ResultsSection';
import { RunForm } from './components/RunForm';
import { countQuestionsLoosely } from './questionsPreview';
import { toCsv } from './csv';
import { downloadBlob } from './download';
import { formatBytes } from './format';
import './styles/global.css';
import type { AnswerResponse, ServerConfig } from './types';
import { oversizeError } from './validation';

export type Phase = 'idle' | 'running' | 'error' | 'success';

function documentMetaText(file: File): string {
  const lower = file.name.toLowerCase();
  const kind = lower.endsWith('.pdf') ? 'PDF' : lower.endsWith('.json') ? 'JSON' : 'file';
  return `${kind} · ${formatBytes(file.size)}`;
}

export default function App() {
  const [questionsFile, setQuestionsFile] = useState<File | null>(null);
  const [documentFile, setDocumentFile] = useState<File | null>(null);
  const [questionsMeta, setQuestionsMeta] = useState<string | null>(null);
  const [phase, setPhase] = useState<Phase>('idle');
  const [error, setError] = useState<ErrorInfo | null>(null);
  const [result, setResult] = useState<AnswerResponse | null>(null);
  const [elapsedMs, setElapsedMs] = useState<number | null>(null);
  const [filter, setFilter] = useState<FilterId>('all');
  const [config, setConfig] = useState<ServerConfig>(DEFAULT_CONFIG);
  const headingRef = useRef<HTMLHeadingElement>(null);

  // The server owns the fallback literal and the size limit; both are
  // configurable, so read them once instead of trusting the compiled defaults.
  useEffect(() => {
    let cancelled = false;
    fetchServerConfig().then((loaded) => {
      if (!cancelled) setConfig(loaded);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  // Best-effort "N questions" preview for the run bar. Parsed client-side
  // purely for display; the server does the real validation on submit.
  useEffect(() => {
    if (!questionsFile) {
      setQuestionsMeta(null);
      return;
    }
    let cancelled = false;
    questionsFile
      .text()
      .then((text) => {
        if (cancelled) return;
        const count = countQuestionsLoosely(text);
        const size = formatBytes(questionsFile.size);
        setQuestionsMeta(count === null ? size : `${count} question${count === 1 ? '' : 's'} · ${size}`);
      })
      .catch(() => {
        if (!cancelled) setQuestionsMeta(formatBytes(questionsFile.size));
      });
    return () => {
      cancelled = true;
    };
  }, [questionsFile]);

  // Move focus to the results heading whenever a run finishes, one way or
  // the other, so screen reader and keyboard users land on the outcome.
  useEffect(() => {
    if (phase === 'error' || phase === 'success') {
      headingRef.current?.focus();
    }
  }, [phase]);

  async function handleSubmit() {
    if (!questionsFile || !documentFile) return;

    const sizeProblem =
      oversizeError(questionsFile, 'questions_file', config.maxFileBytes) ??
      oversizeError(documentFile, 'document_file', config.maxFileBytes);
    if (sizeProblem) {
      setResult(null);
      setError({ status: 413, detail: sizeProblem });
      setPhase('error');
      return;
    }

    setPhase('running');
    setError(null);
    setResult(null);

    const start = performance.now();
    try {
      const response = await submitAnswer(questionsFile, documentFile);
      setElapsedMs(performance.now() - start);
      setResult(response);
      setPhase('success');
    } catch (err) {
      const info: ErrorInfo =
        err instanceof ApiRequestError
          ? { status: err.status, detail: err.message }
          : { status: null, detail: err instanceof Error ? err.message : 'Unknown error' };
      setError(info);
      setPhase('error');
    }
  }

  function handleDownloadJson() {
    if (!result) return;
    downloadBlob(JSON.stringify(result, null, 2), 'answers.json', 'application/json');
  }

  function handleDownloadCsv() {
    if (!result) return;
    downloadBlob(toCsv(result.results), 'answers.csv', 'text/csv');
  }

  return (
    <div className="app">
      <Header modelLine={config.modelLine} />
      <RunForm
        questionsFile={questionsFile}
        documentFile={documentFile}
        questionsMeta={questionsMeta}
        documentMeta={documentFile ? documentMetaText(documentFile) : null}
        isRunning={phase === 'running'}
        submitLabel={result ? 'Run again' : 'Run'}
        onQuestionsFile={setQuestionsFile}
        onDocumentFile={setDocumentFile}
        onSubmit={handleSubmit}
      />
      <ResultsSection
        phase={phase}
        error={error}
        result={result}
        elapsedMs={elapsedMs}
        fallback={config.fallback}
        filter={filter}
        onFilterChange={setFilter}
        headingRef={headingRef}
      />
      <Footer hasResult={result !== null} onDownloadJson={handleDownloadJson} onDownloadCsv={handleDownloadCsv} />
    </div>
  );
}
