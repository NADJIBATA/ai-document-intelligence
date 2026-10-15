import { useRef, useState, type ChangeEvent } from "react";

import { uploadDocument } from "../../api/documents";
import { errorMessage } from "../../lib/errors";
import { Button } from "../ui/Button";
import { UploadIcon } from "../ui/Icons";
import { useToast } from "../ui/toast-context";

const ACCEPTED = ".pdf,.docx,.txt";

interface UploadButtonProps {
  /** Called after each successful upload (refresh the list). */
  onUploaded: () => void;
  variant?: "primary" | "secondary";
}

export function UploadButton({ onUploaded, variant = "primary" }: UploadButtonProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);
  const { notify } = useToast();

  async function onFilesSelected(event: ChangeEvent<HTMLInputElement>): Promise<void> {
    const files = Array.from(event.target.files ?? []);
    event.target.value = ""; // allow selecting the same file again
    if (files.length === 0) {
      return;
    }
    setUploading(true);
    let uploaded = 0;
    for (const file of files) {
      try {
        await uploadDocument(file);
        uploaded += 1;
        onUploaded();
      } catch (error) {
        notify(`${file.name}: ${errorMessage(error)}`, "error");
      }
    }
    setUploading(false);
    if (uploaded > 0) {
      notify(
        uploaded === 1
          ? "Document uploaded. Processing has started."
          : `${uploaded} documents uploaded. Processing has started.`,
        "success",
      );
    }
  }

  return (
    <>
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED}
        multiple
        className="sr-only"
        aria-label="Choose documents to upload"
        tabIndex={-1}
        onChange={(event) => void onFilesSelected(event)}
      />
      <Button
        variant={variant}
        loading={uploading}
        icon={<UploadIcon width={18} height={18} />}
        onClick={() => inputRef.current?.click()}
      >
        {uploading ? "Uploading…" : "Upload Document"}
      </Button>
    </>
  );
}
