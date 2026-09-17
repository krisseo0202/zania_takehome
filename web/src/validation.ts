import { formatBytes } from './format';

/** Client-side pre-check for the API's size limit (published by /health), so
 * an oversized file costs no round trip. Returns null when the file is fine. */
export function oversizeError(file: File, fieldName: string, maxBytes: number): string | null {
  if (file.size <= maxBytes) return null;
  return `${fieldName} is ${formatBytes(file.size)}, over the ${formatBytes(maxBytes)} limit. Choose a smaller file.`;
}


/** Reject by extension, the way the API does. `accept` on the input is only a
 * hint to the file picker and is bypassed entirely by drag and drop. */
export function wrongTypeError(file: File, accepted: string[]): string | null {
  const name = file.name.toLowerCase();
  if (accepted.some((suffix) => name.endsWith(suffix))) return null;
  return `${file.name} is not ${accepted.join(' or ')}.`;
}
