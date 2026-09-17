// Best-effort question count for the run bar's meta line. It only needs to
// recognise the shapes README.md documents; anything else falls back to
// showing just the file size, and the real validation happens server-side.
function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

export function parseQuestionsLoosely(jsonText: string): string[] | null {
  try {
    const parsed: unknown = JSON.parse(jsonText);
    const list = Array.isArray(parsed)
      ? parsed
      : isRecord(parsed) && Array.isArray(parsed.questions)
        ? parsed.questions
        : null;
    if (!list) return null;
    return list.map((item) =>
      typeof item === 'string'
        ? item
        : isRecord(item) && typeof item.question === 'string'
          ? item.question
          : '',
    );
  } catch {
    return null;
  }
}

export function countQuestionsLoosely(jsonText: string): number | null {
  return parseQuestionsLoosely(jsonText)?.length ?? null;
}
