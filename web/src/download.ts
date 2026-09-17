/** Save a string as a file via a throwaway Blob URL, then clean it up. */
export function downloadBlob(content: string, filename: string, mimeType: string): void {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  // Revoking synchronously cancels the download in Safari and some Firefox
  // builds: the fetch of the blob URL has not started when click() returns.
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}
