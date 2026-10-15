import { fetchDocumentFile } from "../api/documents";

/**
 * Open the original file in a new tab. The file endpoint needs the Authorization
 * header, so it is downloaded first and shown from a temporary blob URL.
 * For PDFs, `page` positions the viewer on the cited page.
 */
export async function openDocumentFile(documentId: string, page?: number | null): Promise<void> {
  // Opened synchronously (still inside the click handler), otherwise browsers block it.
  const tab = window.open("", "_blank");
  try {
    const blob = await fetchDocumentFile(documentId);
    const url = URL.createObjectURL(blob);
    const target = blob.type === "application/pdf" && page ? `${url}#page=${page}` : url;
    if (tab) {
      tab.opener = null;
      tab.location.href = target;
    } else {
      window.location.assign(target);
    }
    // Leave time for the viewer to load before releasing the blob.
    window.setTimeout(() => URL.revokeObjectURL(url), 60_000);
  } catch (error) {
    tab?.close();
    throw error;
  }
}

/** Save the original file with its name. */
export async function downloadDocumentFile(documentId: string, filename: string): Promise<void> {
  const blob = await fetchDocumentFile(documentId);
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}
