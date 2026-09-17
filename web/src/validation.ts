import { formatBytes } from './format';

/** Client-side pre-check for the API's size limit (published by /health), so
 * an oversized file costs no round trip. Returns null when the file is fine. */
export function oversizeError(file: File, fieldName: string, maxBytes: number): string | null {
  if (file.size <= maxBytes) return null;
  return `${fieldName} is ${formatBytes(file.size)}, over the ${formatBytes(maxBytes)} limit. Choose a smaller file.`;
}
