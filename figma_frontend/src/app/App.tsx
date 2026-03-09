import { useMemo, useState } from "react";
import { UploadDropzone } from "./components/upload-dropzone";
import { ImageViewerCard } from "./components/image-viewer-card";
import { ProbabilityRow } from "./components/probability-row";
import { LLMCard } from "./components/llm-card";
import { DisclaimerBanner } from "./components/disclaimer-banner";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "./components/ui/card";
import { Button } from "./components/ui/button";
import { Badge } from "./components/ui/badge";
import { Skeleton } from "./components/ui/skeleton";
import {
  Download,
  Filter,
  Image as ImageIcon,
  Search,
  Sparkles,
  ZoomIn,
  ZoomOut,
  Maximize2,
  RotateCcw,
} from "lucide-react";
import { StatusChip } from "./components/status-chip";
import { Input } from "./components/ui/input";

type AppState = "empty" | "loading" | "results";
type Status = "present" | "uncertain" | "not-present";

interface Label {
  label: string;
  probability: number;
  status: Status;
}

interface RankedFinding {
  label: string;
  status: Status;
  probability: number;
  rationale: string;
}

interface Differential {
  condition: string;
  confidence: "low" | "medium" | "high";
  reason: string;
}

interface RecommendedAction {
  action: string;
  urgency: "routine" | "soon" | "urgent";
}

interface ModelOutput {
  labels: Label[];
  llm: {
    summary: string;
    rankedFindings: RankedFinding[];
    differentials: Differential[];
    recommendedActions: RecommendedAction[];
    safetyNote: string;
  };
}

/**
 * NOTE: I slightly edited your mock "recommendedActions" to remove medication advice.
 * That keeps the UI more aligned with "research use only / not medical advice."
 */
const MOCK_OUTPUT: ModelOutput = {
  labels: [
    { label: "Pleural Effusion", probability: 0.92, status: "present" },
    { label: "Cardiomegaly", probability: 0.78, status: "present" },
    { label: "Edema", probability: 0.65, status: "uncertain" },
    { label: "Consolidation", probability: 0.54, status: "uncertain" },
    { label: "Atelectasis", probability: 0.42, status: "uncertain" },
    { label: "Pneumothorax", probability: 0.15, status: "not-present" },
    { label: "Lung Opacity", probability: 0.71, status: "present" },
    { label: "Lung Lesion", probability: 0.12, status: "not-present" },
    { label: "Fracture", probability: 0.08, status: "not-present" },
    { label: "Support Devices", probability: 0.03, status: "not-present" },
    { label: "Pneumonia", probability: 0.48, status: "uncertain" },
    { label: "No Finding", probability: 0.05, status: "not-present" },
    {
      label: "Enlarged Cardiomediastinum",
      probability: 0.68,
      status: "uncertain",
    },
    { label: "Pleural Other", probability: 0.23, status: "not-present" },
  ],
  llm: {
    summary:
      "Findings are most consistent with moderate pleural effusion and cardiomegaly, with additional evidence suggestive of pulmonary edema and possible consolidation. Overall pattern may reflect fluid overload with cardiac involvement.",
    rankedFindings: [
      {
        label: "Pleural Effusion",
        status: "present",
        probability: 0.92,
        rationale:
          "High probability indicates likely pleural fluid accumulation (may be bilateral).",
      },
      {
        label: "Cardiomegaly",
        status: "present",
        probability: 0.78,
        rationale:
          "Enlarged cardiac silhouette can be associated with heart failure or chronic cardiac disease.",
      },
      {
        label: "Lung Opacity",
        status: "present",
        probability: 0.71,
        rationale:
          "Increased opacities may reflect fluid, inflammation, or infection depending on distribution.",
      },
      {
        label: "Enlarged Cardiomediastinum",
        status: "uncertain",
        probability: 0.68,
        rationale:
          "Borderline signal may relate to cardiac enlargement or mediastinal contour changes.",
      },
      {
        label: "Edema",
        status: "uncertain",
        probability: 0.65,
        rationale:
          "Moderate probability supports a possible pulmonary edema pattern.",
      },
    ],
    differentials: [
      {
        condition: "Congestive Heart Failure (CHF)",
        confidence: "high",
        reason:
          "Combination of cardiomegaly, pleural effusion, and possible edema supports decompensated heart failure.",
      },
      {
        condition: "Volume Overload",
        confidence: "high",
        reason:
          "Effusions + edema-like pattern can occur with fluid retention or overload states.",
      },
      {
        condition: "Community-Acquired Pneumonia",
        confidence: "medium",
        reason:
          "Opacity/consolidation signals may reflect infection, though distribution may be less typical.",
      },
      {
        condition: "Renal Dysfunction with Fluid Retention",
        confidence: "medium",
        reason:
          "Fluid overload pattern can be secondary to reduced clearance and volume retention.",
      },
    ],
    recommendedActions: [
      { action: "Compare with prior chest imaging if available", urgency: "soon" },
      {
        action: "Correlate with symptoms, vitals, and relevant labs (e.g., BNP if available)",
        urgency: "soon",
      },
      {
        action: "Consider echocardiography to assess cardiac function (if clinically appropriate)",
        urgency: "routine",
      },
      { action: "Obtain formal radiologist interpretation", urgency: "soon" },
    ],
    safetyNote:
      "AI-generated output for educational/research use only. Not a medical device and not for clinical decision-making.",
  },
};

function App() {
  const [appState, setAppState] = useState<AppState>("empty");
  const [imageUrl, setImageUrl] = useState<string>("");
  const [filterMode, setFilterMode] = useState<"all" | "relevant">("all");
  const [studyId, setStudyId] = useState<string>("XR-2026-0221-001");
  const [zoom, setZoom] = useState(100);

  // Viewer controls are UI-only (non-functional) for now
  const [viewerMode, setViewerMode] = useState<"fit" | "zoom">("fit");

  const handleFileSelect = (file: File) => {
    const url = URL.createObjectURL(file);
    setImageUrl(url);
    setAppState("loading");

    // Simulate processing
    setTimeout(() => {
      setAppState("results");
    }, 1800);
  };

  const handleUseSample = () => {
    // For a mock: you can point this to a static asset later.
    // For now: just transition to "results" with a placeholder
    setImageUrl(""); // keep empty or set to "/sample-xray.png" if you add an asset
    setAppState("loading");
    setTimeout(() => setAppState("results"), 900);
  };

  const handleReset = () => {
    setAppState("empty");
    setImageUrl("");
    setFilterMode("all");
    setViewerMode("fit");
    setZoom(100);
  };

  const handleZoomIn = () => setZoom((z) => Math.min(z + 25, 500));
  const handleZoomOut = () => setZoom((z) => Math.max(z - 25, 10));
  const handleFit = () => {
    setZoom(100);
    setViewerMode("fit");
  };
  const handleZoomReset = () => {
    setZoom(100);
  };
  const handleZoomToFit = () => {
    setZoom(50); // Zoom out to see full image
  };

  const handleExportJSON = () => {
    const exportData = {
      studyId,
      timestamp: new Date().toISOString(),
      modelOutput: MOCK_OUTPUT,
    };
    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `xray-analysis-${studyId}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleDownloadReport = () => {
    // Create a simple text report
    let report = `CHEST X-RAY AI ANALYSIS REPORT\n`;
    report += `${"=".repeat(50)}\n\n`;
    report += `Study ID: ${studyId}\n`;
    report += `Date: ${new Date().toLocaleDateString()}\n`;
    report += `Time: ${new Date().toLocaleTimeString()}\n\n`;
    report += `DISCLAIMER: ${MOCK_OUTPUT.llm.safetyNote}\n\n`;
    report += `${"=".repeat(50)}\n\n`;
    report += `SUMMARY:\n${MOCK_OUTPUT.llm.summary}\n\n`;
    report += `${"=".repeat(50)}\n\n`;
    report += `MODEL PREDICTIONS (CheXpert-14):\n`;
    sortedLabels.forEach((label) => {
      report += `- ${label.label}: ${(label.probability * 100).toFixed(1)}% (${label.status})\n`;
    });
    report += `\n${"=".repeat(50)}\n\n`;
    report += `RANKED FINDINGS:\n`;
    MOCK_OUTPUT.llm.rankedFindings.forEach((finding, idx) => {
      report += `${idx + 1}. ${finding.label} - ${finding.status} (${(finding.probability * 100).toFixed(1)}%)\n`;
      report += `   ${finding.rationale}\n\n`;
    });
    report += `${"=".repeat(50)}\n\n`;
    report += `POSSIBLE DIFFERENTIALS:\n`;
    MOCK_OUTPUT.llm.differentials.forEach((diff, idx) => {
      report += `${idx + 1}. ${diff.condition} (Confidence: ${diff.confidence})\n`;
      report += `   ${diff.reason}\n\n`;
    });
    report += `${"=".repeat(50)}\n\n`;
    report += `RECOMMENDED NEXT STEPS:\n`;
    MOCK_OUTPUT.llm.recommendedActions.forEach((action, idx) => {
      report += `${idx + 1}. [${action.urgency.toUpperCase()}] ${action.action}\n`;
    });

    const blob = new Blob([report], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `xray-report-${studyId}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const sortedLabels = useMemo(() => {
    const filtered =
      filterMode === "relevant"
        ? MOCK_OUTPUT.labels.filter((l) => l.status !== "not-present")
        : MOCK_OUTPUT.labels;
    return [...filtered].sort((a, b) => b.probability - a.probability);
  }, [filterMode]);

  const topFindings = useMemo(() => {
    // Show top 3 "present/uncertain" items for chips
    const relevant = [...MOCK_OUTPUT.labels]
      .filter((l) => l.status !== "not-present")
      .sort((a, b) => b.probability - a.probability)
      .slice(0, 3);
    return relevant;
  }, []);

  const statusLegend = (
    <div className="flex flex-wrap gap-2 text-xs text-gray-600">
      <span className="flex items-center gap-2">
        <StatusChip status="present" /> Present ≥ 0.70
      </span>
      <span className="flex items-center gap-2">
        <StatusChip status="uncertain" /> Uncertain 0.30–0.69
      </span>
      <span className="flex items-center gap-2">
        <StatusChip status="not-present" /> Not &lt; 0.30
      </span>
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b">
        <div className="max-w-[1440px] mx-auto px-6 py-3">
          <div className="flex items-start justify-between gap-6">
            <div>
              <h1 className="text-xl font-semibold tracking-tight">
                Chest X-ray AI Analysis
              </h1>
              <p className="text-xs text-gray-600 mt-0.5">
                Vision model → CheXpert-14 → Local LLM (Ollama) explanation
              </p>

              {/* Small "mode" badges make it feel product-y */}
              <div className="flex gap-2 mt-2">
                <Badge variant="secondary" className="bg-gray-100 text-gray-700 text-xs">
                  Research Demo
                </Badge>
                <Badge variant="secondary" className="bg-gray-100 text-gray-700 text-xs">
                  Local Inference
                </Badge>
                <Badge variant="secondary" className="bg-gray-100 text-gray-700 text-xs">
                  JSON Output
                </Badge>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <Button variant="outline" size="sm" disabled={appState !== "results"} className="h-8 text-xs" onClick={handleExportJSON}>
                <Download className="w-3 h-3 mr-1" />
                Export JSON
              </Button>
              <Button variant="outline" size="sm" disabled={appState !== "results"} className="h-8 text-xs" onClick={handleDownloadReport}>
                <Download className="w-3 h-3 mr-1" />
                Download Report
              </Button>
              {appState !== "empty" && (
                <>
                  <div className="h-6 w-px bg-gray-300 mx-1" />
                  <div className="flex items-center gap-2 px-3 py-1 bg-gray-100 rounded-md">
                    <span className="text-xs text-gray-600">Study ID:</span>
                    <span className="text-xs font-medium text-gray-900">{studyId}</span>
                  </div>
                </>
              )}
              {appState !== "empty" && (
                <Button variant="outline" size="sm" onClick={handleReset} className="h-8 text-xs">
                  New Analysis
                </Button>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main */}
      <div className="max-w-[1440px] mx-auto p-4">
        <div className="grid grid-cols-3 gap-3">
          {/* LEFT: Upload + Viewer */}
          <div className="space-y-3">
            {appState === "empty" && (
              <>
                {/* Polished upload container + "Use sample" */}
                <Card className="border-dashed">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm flex items-center gap-2">
                      <ImageIcon className="w-4 h-4 text-gray-600" />
                      Upload X-ray
                    </CardTitle>
                    <CardDescription className="text-xs">
                      PNG/JPG. De-identified only.
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-2">
                    <UploadDropzone onFileSelect={handleFileSelect} />
                    <div className="flex items-center gap-2">
                      <Button size="sm" onClick={handleUseSample}>
                        Use sample
                      </Button>
                      <span className="text-xs text-gray-500 ml-auto">
                        Max 20MB
                      </span>
                    </div>
                  </CardContent>
                </Card>

                {/* Study ID Input */}
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm">Study ID</CardTitle>
                    <CardDescription className="text-xs">
                      Enter a unique identifier for this X-ray
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <Input
                      type="text"
                      value={studyId}
                      onChange={(e) => setStudyId(e.target.value)}
                      placeholder="XR-YYYY-MMDD-###"
                      className="text-xs h-8"
                    />
                  </CardContent>
                </Card>

                {/* Placeholder viewer */}
                <Card className="bg-gradient-to-b from-gray-50 to-gray-100">
                  <CardContent className="p-8">
                    <div className="flex flex-col items-center justify-center text-center gap-2">
                      <div className="w-8 h-8 rounded-full bg-white border flex items-center justify-center">
                        <Search className="w-4 h-4 text-gray-500" />
                      </div>
                      <p className="text-xs text-gray-700 font-medium">
                        X-ray preview
                      </p>
                      <p className="text-xs text-gray-500">
                        Upload to view
                      </p>
                    </div>
                  </CardContent>
                </Card>
              </>
            )}

            {(appState === "loading" || appState === "results") && (
              <>
                <div className="flex items-center justify-between">
                  <h2 className="text-sm font-semibold">X-ray Viewer</h2>
                  <Button variant="ghost" size="sm" onClick={handleReset} className="h-7 text-xs">
                    New
                  </Button>
                </div>

                {/* Viewer toolbar mock (can wire later) */}
                <Card>
                  <CardHeader className="pb-2">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-1">
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={handleZoomToFit}
                          className="h-7 text-xs px-2"
                        >
                          Fit to View
                        </Button>
                        <div className="h-4 w-px bg-gray-300 mx-1" />
                        <Button
                          size="icon"
                          variant="outline"
                          onClick={handleFit}
                          aria-label="Fit"
                          className="h-7 w-7"
                        >
                          <Maximize2 className="w-3 h-3" />
                        </Button>
                        <Button size="icon" variant="outline" aria-label="Zoom in" className="h-7 w-7" onClick={handleZoomIn}>
                          <ZoomIn className="w-3 h-3" />
                        </Button>
                        <Button size="icon" variant="outline" aria-label="Zoom out" className="h-7 w-7" onClick={handleZoomOut}>
                          <ZoomOut className="w-3 h-3" />
                        </Button>
                        <Button size="icon" variant="outline" aria-label="Reset" className="h-7 w-7" onClick={handleZoomReset}>
                          <RotateCcw className="w-3 h-3" />
                        </Button>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent className="pb-3">
                    <ImageViewerCard 
                      imageUrl={imageUrl} 
                      studyId={studyId}
                      zoom={zoom} 
                      onZoomIn={handleZoomIn}
                      onZoomOut={handleZoomOut}
                      onFit={handleFit}
                      onReset={handleZoomReset}
                    />
                  </CardContent>
                </Card>
              </>
            )}
          </div>

          {/* MIDDLE: Predictions */}
          <div className="space-y-3">
            <Card className="shadow-sm">
              <CardHeader className="pb-3">
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <CardTitle className="text-sm">Model Predictions</CardTitle>
                    <CardDescription className="text-xs">
                      CheXpert 14-label probabilities
                    </CardDescription>

                    {/* Top finding chips (only show when results) */}
                    {appState === "results" && (
                      <div className="flex flex-wrap gap-1 mt-2">
                        {topFindings.map((t) => (
                          <Badge
                            key={t.label}
                            variant="secondary"
                            className="bg-gray-100 text-gray-800 text-xs"
                          >
                            {t.label} • {(t.probability * 100).toFixed(0)}%
                          </Badge>
                        ))}
                      </div>
                    )}
                  </div>

                  {appState === "results" && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() =>
                        setFilterMode(filterMode === "all" ? "relevant" : "all")
                      }
                      className="h-7 text-xs"
                    >
                      <Filter className="w-3 h-3 mr-1" />
                      {filterMode === "all" ? "Relevant" : "All"}
                    </Button>
                  )}
                </div>
              </CardHeader>

              <CardContent className="space-y-2 pt-0">
                {appState === "empty" && (
                  <div className="py-8 text-center">
                    <p className="text-xs text-gray-600">
                      Upload an X-ray to generate predictions.
                    </p>
                  </div>
                )}

                {appState === "loading" && (
                  <div className="space-y-2">
                    <div className="flex items-center gap-2 text-xs text-gray-700">
                      <Sparkles className="w-3 h-3 text-gray-500" />
                      Computing probabilities…
                    </div>
                    {[...Array(8)].map((_, i) => (
                      <div key={i} className="flex items-center gap-2">
                        <Skeleton className="h-3 flex-1" />
                        <Skeleton className="h-3 w-16" />
                        <Skeleton className="h-5 w-20" />
                      </div>
                    ))}
                  </div>
                )}

                {appState === "results" && (
                  <div className="overflow-x-auto max-h-[calc(100vh-280px)] overflow-y-auto">
                    {/* Legend */}
                    <div className="flex flex-col gap-1 text-xs text-gray-600 pb-2 min-w-max">
                      <span className="flex items-center gap-1">
                        <StatusChip status="present" /> Present ≥ 0.70
                      </span>
                      <span className="flex items-center gap-1">
                        <StatusChip status="uncertain" /> Uncertain 0.30–0.69
                      </span>
                      <span className="flex items-center gap-1">
                        <StatusChip status="not-present" /> Not present &lt; 0.30
                      </span>
                    </div>

                    {/* Rows */}
                    <div className="border-t pt-2 min-w-max">
                      {sortedLabels.map((item, idx) => (
                        <ProbabilityRow
                          key={`${item.label}-${idx}`}
                          label={item.label}
                          probability={item.probability}
                          status={item.status}
                        />
                      ))}
                    </div>

                    <p className="text-xs text-gray-500 mt-2 pt-2 border-t min-w-max">
                      These scores drive the LLM explanation.
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* COLUMN 3: LLM Summary + Ranked Findings */}
          <div className="space-y-3">
            {appState === "empty" && (
              <Card className="bg-gradient-to-b from-gray-50 to-gray-100">
                <CardContent className="p-8">
                  <div className="flex flex-col items-center justify-center text-center gap-2">
                    <div className="w-8 h-8 rounded-full bg-white border flex items-center justify-center">
                      <Sparkles className="w-4 h-4 text-gray-500" />
                    </div>
                    <p className="text-xs text-gray-700 font-medium">
                      AI findings
                    </p>
                    <p className="text-xs text-gray-500">
                      LLM summary appears here
                    </p>
                  </div>
                </CardContent>
              </Card>
            )}

            {appState === "loading" && (
              <div className="space-y-3">
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm">AI Findings</CardTitle>
                    <CardDescription className="text-xs">Generating…</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-2">
                    <Skeleton className="h-3 w-full" />
                    <Skeleton className="h-3 w-full" />
                    <Skeleton className="h-3 w-3/4" />
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm">Ranked Findings</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2">
                    {[...Array(3)].map((_, i) => (
                      <div key={i} className="p-2 border rounded">
                        <Skeleton className="h-3 w-full mb-1" />
                        <Skeleton className="h-3 w-2/3" />
                      </div>
                    ))}
                  </CardContent>
                </Card>
              </div>
            )}

            {appState === "results" && (
              <div className="space-y-3">
                {/* Sticky safety note */}
                <Card className="border-amber-200 bg-amber-50">
                  <CardContent className="py-3">
                    <div className="flex flex-col gap-2">
                      <div className="flex items-start gap-2">
                        <Badge className="bg-amber-100 text-amber-900 text-xs">
                          Research Only
                        </Badge>
                        <div className="flex-1">
                          <p className="text-xs text-amber-900 leading-relaxed">
                            AI output for education only. Not for clinical use.
                          </p>
                          <p className="text-xs text-amber-800 leading-relaxed mt-1">
                            For research and education only. Not for clinical decision-making.
                          </p>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <LLMCard
                  title="AI Findings Summary"
                  description="Generated from model outputs"
                >
                  <p className="text-xs leading-relaxed">{MOCK_OUTPUT.llm.summary}</p>
                </LLMCard>

                <LLMCard title="Ranked Findings">
                  <div className="space-y-2">
                    {MOCK_OUTPUT.llm.rankedFindings.map((finding, idx) => (
                      <div key={idx} className="p-2 bg-gray-50 rounded border overflow-hidden">
                        <div className="flex items-start justify-between mb-1 gap-2">
                          <span className="text-xs font-medium text-gray-800 flex-1">
                            {finding.label}
                          </span>
                          <div className="flex items-center gap-1 flex-shrink-0">
                            <span className="text-xs text-gray-600 tabular-nums">
                              {(finding.probability * 100).toFixed(1)}%
                            </span>
                            <StatusChip status={finding.status} />
                          </div>
                        </div>
                        <p className="text-xs text-gray-600 leading-relaxed">{finding.rationale}</p>
                      </div>
                    ))}
                  </div>
                </LLMCard>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;