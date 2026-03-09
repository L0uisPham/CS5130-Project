import { Upload } from "lucide-react";
import { Card } from "./ui/card";

interface UploadDropzoneProps {
  onFileSelect: (file: File) => void;
}

export function UploadDropzone({ onFileSelect }: UploadDropzoneProps) {
  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && (file.type.startsWith("image/") || file.name.endsWith(".dicom"))) {
      onFileSelect(file);
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onFileSelect(file);
    }
  };

  return (
    <Card
      className="border-2 border-dashed border-gray-300 bg-gray-50 hover:border-gray-400 transition-colors cursor-pointer"
      onDrop={handleDrop}
      onDragOver={(e) => e.preventDefault()}
    >
      <label className="flex flex-col items-center justify-center p-6 cursor-pointer">
        <Upload className="w-8 h-8 text-gray-400 mb-2" />
        <p className="text-xs text-gray-700 mb-1">
          Drag & drop or Browse
        </p>
        <p className="text-xs text-gray-500">De-identified data only</p>
        <input
          type="file"
          className="hidden"
          accept="image/*,.dicom"
          onChange={handleFileInput}
        />
      </label>
    </Card>
  );
}