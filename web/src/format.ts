/** Human-readable file size, e.g. 2.1 KB, 6.5 MB. */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ['KB', 'MB', 'GB'];
  let value = bytes / 1024;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(1)} ${units[unitIndex]}`;
}

/** Human-readable wall-clock duration, e.g. 420 ms, 6.4 s. */
export function formatDuration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(1)} s`;
}

/** Small dollar amounts without rounding a real cost to $0.00. */
export function formatUsd(dollars: number): string {
  if (dollars > 0 && dollars < 0.001) return '<$0.001';
  return `$${dollars.toFixed(3)}`;
}

/** `page:14` -> `p. 14`, `json:hosting` -> `hosting`. */
export function formatSource(sourceId: string): string {
  const [kind, ...rest] = sourceId.split(':');
  const label = rest.join(':');
  return kind === 'page' ? `p. ${label}` : label || sourceId;
}

/** The best-matching source, with a count of the others retrieved. */
export function formatSources(sources: string[] | undefined): string {
  if (!sources || sources.length === 0) return '—';
  const [first, ...others] = sources;
  return others.length ? `${formatSource(first)} +${others.length}` : formatSource(first);
}
