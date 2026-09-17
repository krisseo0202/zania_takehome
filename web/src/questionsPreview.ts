// Best-effort question count for the run bar's meta line. It only needs to
// recognise the shapes README.md documents; anything else falls back to
// showing just the file size, and the real validation happens server-side.
function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

export function countQuestionsLoosely(jsonText: string): number | null {
  try {
    const parsed: unknown = JSON.parse(jsonText);
    if (Array.isArray(parsed)) return parsed.length;
    if (isRecord(parsed) && Array.isArray(parsed.questions)) return parsed.questions.length;
    return null;
  } catch {
    return null;
  }
}
