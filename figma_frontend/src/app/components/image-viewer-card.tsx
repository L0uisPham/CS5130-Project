import { useState } from "react";
import { ZoomIn, ZoomOut, Maximize2, RotateCcw, Move } from "lucide-react";
import { Card } from "./ui/card";
import { Button } from "./ui/button";
import { Switch } from "./ui/switch";
import { Label } from "./ui/label";

interface ImageViewerCardProps {
  imageUrl: string;
  studyId: string;
  zoom: number;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onFit: () => void;
  onReset: () => void;
}

export function ImageViewerCard({ 
  imageUrl, 
  studyId,
  zoom,
  onZoomIn,
  onZoomOut,
  onFit,
  onReset
}: ImageViewerCardProps) {
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsPanning(true);
    setPanStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isPanning) {
      setPan({
        x: e.clientX - panStart.x,
        y: e.clientY - panStart.y,
      });
    }
  };

  const handleMouseUp = () => {
    setIsPanning(false);
  };

  const handleResetLocal = () => {
    setPan({ x: 0, y: 0 });
    setShowHeatmap(false);
    onReset();
  };

  return (
    <Card className="overflow-hidden">
      <div
        className="bg-slate-800 relative overflow-hidden"
        style={{ height: "320px", cursor: isPanning ? "grabbing" : "grab" }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        <img
          src={imageUrl}
          alt="X-ray"
          className="absolute top-1/2 left-1/2 max-w-none select-none"
          style={{
            transform: `translate(-50%, -50%) translate(${pan.x}px, ${pan.y}px) scale(${zoom / 100})`,
            transition: isPanning ? "none" : "transform 0.2s ease-out",
          }}
          draggable={false}
        />
        {showHeatmap && (
          <div
            className="absolute top-1/2 left-1/2 pointer-events-none"
            style={{
              transform: `translate(-50%, -50%) translate(${pan.x}px, ${pan.y}px) scale(${zoom / 100})`,
              width: "300px",
              height: "300px",
              background:
                "radial-gradient(circle, rgba(255,0,0,0.3) 0%, rgba(255,255,0,0.2) 50%, transparent 70%)",
            }}
          />
        )}
        <div className="absolute top-2 right-2 bg-black/50 text-white px-2 py-1 rounded text-xs">
          {zoom}%
        </div>
      </div>
      <div className="p-2 bg-gray-50 border-t">
        <div className="flex gap-3 text-xs text-gray-600">
          <span>ID: {studyId}</span>
          <span>PA</span>
          <span>2026-02-21</span>
        </div>
      </div>
    </Card>
  );
}