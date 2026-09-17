import type { AnswerResultItem } from './types';

// A field starting with = + - @ (or tab/CR, which Excel strips to reach them)
// is executed as a formula on open. Answers are model output derived from an
// uploaded document, so they are attacker-influenceable: neutralise with a
// leading apostrophe, which spreadsheets treat as "this is text".
function neutralizeFormula(value: string): string {
  return /^[=+\-@\t\r]/.test(value) ? `'${value}` : value;
}

// RFC 4180: quote a field if it contains a comma, quote, or line break, and
// double any quote characters inside it.
function csvField(value: string): string {
  const safe = neutralizeFormula(value);
  const needsQuoting = /[",\r\n]/.test(safe);
  const escaped = safe.replace(/"/g, '""');
  return needsQuoting ? `"${escaped}"` : escaped;
}

export function toCsv(results: AnswerResultItem[]): string {
  const lines = ['question,answer'];
  for (const { question, answer } of results) {
    lines.push(`${csvField(question)},${csvField(answer)}`);
  }
  return lines.join('\r\n');
}
