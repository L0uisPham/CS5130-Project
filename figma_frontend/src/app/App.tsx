import { useEffect, useMemo, useRef, useState } from "react";
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
  X,
  ZoomIn,
  ZoomOut,
  Maximize2,
  RotateCcw,
} from "lucide-react";
import { StatusChip } from "./components/status-chip";
import { Input } from "./components/ui/input";

type AppState = "empty" | "loading" | "results" | "error";
type Status = "present" | "uncertain" | "not-present";
type LLMState = "idle" | "loading" | "ready" | "error";

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
  modelUsed?: string;
  labels: Label[];
  llm: {
    summary: string;
    rankedFindings: RankedFinding[];
    differentials: Differential[];
    recommendedActions: RecommendedAction[];
    safetyNote: string;
  };
  rawCsv?: string;
}

interface LLMOutput {
  summary: string;
  rankedFindings: RankedFinding[];
  differentials: Differential[];
  recommendedActions: RecommendedAction[];
  safetyNote: string;
}

interface ModelOption {
  value: string;
  label: string;
}

interface RunHistoryEntry {
  id: string;
  studyId: string;
  createdAt: string;
  imageDataUrl: string;
  modelOutput: ModelOutput;
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
  rawCsv: [
    "label,probability,status",
    "Pleural Effusion,0.9200,present",
    "Cardiomegaly,0.7800,present",
    "Lung Opacity,0.7100,present",
    "Enlarged Cardiomediastinum,0.6800,uncertain",
    "Edema,0.6500,uncertain",
    "Consolidation,0.5400,uncertain",
    "Pneumonia,0.4800,uncertain",
    "Atelectasis,0.4200,uncertain",
    "Pleural Other,0.2300,not-present",
    "Pneumothorax,0.1500,not-present",
    "Lung Lesion,0.1200,not-present",
    "Fracture,0.0800,not-present",
    "No Finding,0.0500,not-present",
    "Support Devices,0.0300,not-present",
  ].join("\n"),
};

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "") ?? "http://localhost:8001";

const FALLBACK_MODELS: ModelOption[] = [
  { value: "convnext_t", label: "ConvNeXt Tiny" },
  { value: "swin_tiny", label: "Swin Tiny" },
  { value: "ensemble", label: "ConvNeXt + Swin Ensemble" },
];

const EMPTY_LLM_OUTPUT: LLMOutput = {
  summary: "",
  rankedFindings: [],
  differentials: [],
  recommendedActions: [],
  safetyNote: "",
};

const RUN_HISTORY_STORAGE_KEY = "chest-xray-run-history";
const MAX_RUN_HISTORY = 20;

async function fileToDataUrl(file: File): Promise<string> {
  return await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === "string") {
        resolve(reader.result);
        return;
      }
      reject(new Error("Could not convert image to data URL."));
    };
    reader.onerror = () => reject(new Error("Could not read image file."));
    reader.readAsDataURL(file);
  });
}

function formatRunTimestamp(timestamp: string): string {
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) {
    return timestamp;
  }
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(date);
}

function App() {
  const [appState, setAppState] = useState<AppState>("empty");
  const [llmState, setLlmState] = useState<LLMState>("idle");
  const [imageUrl, setImageUrl] = useState<string>("");
  const [modelOutput, setModelOutput] = useState<ModelOutput | null>(null);
  const [errorMessage, setErrorMessage] = useState<string>("");
  const [llmErrorMessage, setLlmErrorMessage] = useState<string>("");
  const [filterMode, setFilterMode] = useState<"all" | "relevant">("all");
  const [studyId, setStudyId] = useState<string>("XR-2026-0221-001");
  const [zoom, setZoom] = useState(100);
  const [models, setModels] = useState<ModelOption[]>(FALLBACK_MODELS);
  const [selectedModel, setSelectedModel] = useState<string>(FALLBACK_MODELS[0].value);
  const [runHistory, setRunHistory] = useState<RunHistoryEntry[]>([]);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const analysisRequestIdRef = useRef(0);
  const pendingHistoryRef = useRef<{
    requestId: number;
    id: string;
    createdAt: string;
    studyId: string;
    imageDataUrl: string;
  } | null>(null);

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(RUN_HISTORY_STORAGE_KEY);
      if (!raw) {
        return;
      }
      const parsed = JSON.parse(raw) as RunHistoryEntry[];
      if (Array.isArray(parsed)) {
        setRunHistory(parsed);
      }
    } catch {
      setRunHistory([]);
    }
  }, []);

  useEffect(() => {
    window.localStorage.setItem(RUN_HISTORY_STORAGE_KEY, JSON.stringify(runHistory));
  }, [runHistory]);

  useEffect(() => {
    let cancelled = false;

    const loadModels = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/models`);
        if (!response.ok) {
          throw new Error(`Model lookup failed with status ${response.status}`);
        }

        const payload = (await response.json()) as {
          models?: ModelOption[];
          defaultModel?: string;
        };

        if (cancelled || !payload.models?.length) {
          return;
        }

        setModels(payload.models);
        setSelectedModel(payload.defaultModel ?? payload.models[0].value);
      } catch {
        if (!cancelled) {
          setModels(FALLBACK_MODELS);
        }
      }
    };

    void loadModels();

    return () => {
      cancelled = true;
    };
  }, []);

  const saveRunToHistory = (entry: RunHistoryEntry) => {
    setRunHistory((current) => [
      entry,
      ...current.filter((run) => run.id !== entry.id).slice(0, MAX_RUN_HISTORY - 1),
    ]);
    setActiveRunId(entry.id);
  };

  const analyzeFile = async (file: File, imageDataUrl: string) => {
    const requestId = ++analysisRequestIdRef.current;
    const createdAt = new Date().toISOString();
    setErrorMessage("");
    setLlmErrorMessage("");
    setLlmState("idle");
    setModelOutput(null);
    setAppState("loading");
    setActiveRunId(null);
    pendingHistoryRef.current = {
      requestId,
      id: `${studyId}-${createdAt}`,
      createdAt,
      studyId,
      imageDataUrl,
    };

    const formData = new FormData();
    formData.append("file", file);
    formData.append("model_name", selectedModel);

    try {
      const response = await fetch(`${API_BASE_URL}/analyze`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const fallbackMessage = `Analysis failed with status ${response.status}`;
        try {
          const detail = (await response.json())?.detail;
          throw new Error(typeof detail === "string" ? detail : fallbackMessage);
        } catch (error) {
          if (error instanceof Error) {
            throw error;
          }
          throw new Error(fallbackMessage);
        }
      }

      const output = (await response.json()) as Omit<ModelOutput, "llm">;
      const labelsForLLM = output.labels.map((label) => label.label);
      const probsForLLM = output.labels.map((label) => label.probability);

      if (analysisRequestIdRef.current !== requestId) {
        return;
      }

      setModelOutput({
        ...output,
        llm: EMPTY_LLM_OUTPUT,
      });
      setAppState("results");
      setLlmState("loading");

      void (async () => {
        try {
          const llmResponse = await fetch(`${API_BASE_URL}/explain`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              labels: labelsForLLM,
              probs: probsForLLM,
            }),
          });

          if (!llmResponse.ok) {
            const fallbackMessage = `LLM generation failed with status ${llmResponse.status}`;
            try {
              const detail = (await llmResponse.json())?.detail;
              throw new Error(typeof detail === "string" ? detail : fallbackMessage);
            } catch (error) {
              if (error instanceof Error) {
                throw error;
              }
              throw new Error(fallbackMessage);
            }
          }

          const llm = (await llmResponse.json()) as LLMOutput;
          if (analysisRequestIdRef.current !== requestId) {
            return;
          }
          setModelOutput((current) => {
            if (!current) {
              return current;
            }
            const updatedOutput = {
              ...current,
              llm,
            };
            const pending = pendingHistoryRef.current;
            if (pending && pending.requestId === requestId) {
              saveRunToHistory({
                id: pending.id,
                studyId: pending.studyId,
                createdAt: pending.createdAt,
                imageDataUrl: pending.imageDataUrl,
                modelOutput: updatedOutput,
              });
              pendingHistoryRef.current = null;
            }
            return updatedOutput;
          });
          setLlmState("ready");
        } catch (error) {
          if (analysisRequestIdRef.current !== requestId) {
            return;
          }
          setLlmErrorMessage(
            error instanceof Error
              ? error.message
              : "The frontend could not reach the explanation API.",
          );
          setLlmState("error");
        }
      })();
    } catch (error) {
      pendingHistoryRef.current = null;
      const message =
        error instanceof Error
          ? error.message
          : "The frontend could not reach the analysis API.";
      setErrorMessage(message);
      setAppState("error");
    }
  };

  const handleFileSelect = (file: File) => {
    void (async () => {
      try {
        const dataUrl = await fileToDataUrl(file);
        setImageUrl(dataUrl);
        void analyzeFile(file, dataUrl);
      } catch (error) {
        setErrorMessage(
          error instanceof Error ? error.message : "Could not read the selected image.",
        );
        setAppState("error");
      }
    })();
  };

  const handleUseSample = () => {
    analysisRequestIdRef.current += 1;
    pendingHistoryRef.current = null;
    const sampleOutput = {
      ...MOCK_OUTPUT,
      modelUsed: selectedModel,
    };
    setImageUrl("");
    setErrorMessage("");
    setLlmErrorMessage("");
    setModelOutput(sampleOutput);
    setAppState("results");
    setLlmState("ready");
    saveRunToHistory({
      id: `${studyId}-${Date.now()}`,
      studyId,
      createdAt: new Date().toISOString(),
      imageDataUrl: "",
      modelOutput: sampleOutput,
    });
  };

  const handleSelectRun = (run: RunHistoryEntry) => {
    analysisRequestIdRef.current += 1;
    pendingHistoryRef.current = null;
    setActiveRunId(run.id);
    setStudyId(run.studyId);
    setSelectedModel(run.modelOutput.modelUsed ?? selectedModel);
    setImageUrl(run.imageDataUrl);
    setModelOutput(run.modelOutput);
    setAppState("results");
    setLlmState("ready");
    setErrorMessage("");
    setLlmErrorMessage("");
    setFilterMode("all");
    setZoom(100);
  };

  const handleDeleteRun = (runId: string) => {
    setRunHistory((current) => current.filter((run) => run.id !== runId));
    setActiveRunId((current) => (current === runId ? null : current));
  };

  const handleReset = () => {
    analysisRequestIdRef.current += 1;
    pendingHistoryRef.current = null;
    setAppState("empty");
    setLlmState("idle");
    setImageUrl("");
    setModelOutput(null);
    setErrorMessage("");
    setLlmErrorMessage("");
    setFilterMode("all");
    setZoom(100);
  };

  const handleZoomIn = () => setZoom((z) => Math.min(z + 25, 500));
  const handleZoomOut = () => setZoom((z) => Math.max(z - 25, 10));
  const handleFit = () => {
    setZoom(100);
  };
  const handleZoomReset = () => {
    setZoom(100);
  };
  const handleZoomToFit = () => {
    setZoom(50); // Zoom out to see full image
  };

  const handleExportJSON = () => {
    if (!modelOutput) {
      return;
    }
    const exportData = {
      studyId,
      timestamp: new Date().toISOString(),
      modelOutput,
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
    if (!modelOutput) {
      return;
    }
    // Create a simple text report
    let report = `CHEST X-RAY AI ANALYSIS REPORT\n`;
    report += `${"=".repeat(50)}\n\n`;
    report += `Study ID: ${studyId}\n`;
    report += `Date: ${new Date().toLocaleDateString()}\n`;
    report += `Time: ${new Date().toLocaleTimeString()}\n\n`;
    report += `DISCLAIMER: ${modelOutput.llm.safetyNote}\n\n`;
    report += `${"=".repeat(50)}\n\n`;
    report += `SUMMARY:\n${modelOutput.llm.summary}\n\n`;
    report += `${"=".repeat(50)}\n\n`;
    report += `MODEL PREDICTIONS (CheXpert-14):\n`;
    sortedLabels.forEach((label) => {
      report += `- ${label.label}: ${(label.probability * 100).toFixed(1)}% (${label.status})\n`;
    });
    report += `\n${"=".repeat(50)}\n\n`;
    report += `RANKED FINDINGS:\n`;
    modelOutput.llm.rankedFindings.forEach((finding, idx) => {
      report += `${idx + 1}. ${finding.label} - ${finding.status} (${(finding.probability * 100).toFixed(1)}%)\n`;
      report += `   ${finding.rationale}\n\n`;
    });
    report += `${"=".repeat(50)}\n\n`;
    report += `POSSIBLE DIFFERENTIALS:\n`;
    modelOutput.llm.differentials.forEach((diff, idx) => {
      report += `${idx + 1}. ${diff.condition} (Confidence: ${diff.confidence})\n`;
      report += `   ${diff.reason}\n\n`;
    });
    report += `${"=".repeat(50)}\n\n`;
    report += `RECOMMENDED NEXT STEPS:\n`;
    modelOutput.llm.recommendedActions.forEach((action, idx) => {
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
    if (!modelOutput) {
      return [];
    }
    const filtered =
      filterMode === "relevant"
        ? modelOutput.labels.filter((l) => l.status !== "not-present")
        : modelOutput.labels;
    return [...filtered].sort((a, b) => b.probability - a.probability);
  }, [filterMode, modelOutput]);

  const topFindings = useMemo(() => {
    if (!modelOutput) {
      return [];
    }
    // Show top 3 "present/uncertain" items for chips
    const relevant = [...modelOutput.labels]
      .filter((l) => l.status !== "not-present")
      .sort((a, b) => b.probability - a.probability)
      .slice(0, 3);
    return relevant;
  }, [modelOutput]);

  const footerDisclaimer =
    modelOutput?.llm.safetyNote ||
    "For research and education only. Not for clinical decision-making.";

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      {/* Header */}
      <header className="bg-white border-b">
        <div className="max-w-[1440px] mx-auto px-6 py-3">
          <div className="flex items-start justify-between gap-6">
            <div>
              <h1 className="text-xl font-semibold tracking-tight">
                Chest X-ray Analysis
              </h1>
            </div>

            <div className="flex items-center gap-2">
              <Button variant="outline" size="sm" disabled={appState !== "results" || !modelOutput} className="h-8 text-xs" onClick={handleExportJSON}>
                <Download className="w-3 h-3 mr-1" />
                Export JSON
              </Button>
              <Button variant="outline" size="sm" disabled={appState !== "results" || !modelOutput || llmState !== "ready"} className="h-8 text-xs" onClick={handleDownloadReport}>
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
      <div className="max-w-[1440px] mx-auto p-4 flex-1 w-full">
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
                      PNG/JPG. Pick a local checkpoint, then upload a de-identified study.
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-2">
                    <div className="space-y-1">
                      <label className="text-xs font-medium text-gray-700" htmlFor="model-select">
                        Inference model
                      </label>
                      <select
                        id="model-select"
                        value={selectedModel}
                        onChange={(e) => setSelectedModel(e.target.value)}
                        className="h-9 w-full rounded-md border border-gray-200 bg-white px-3 text-sm text-gray-900 outline-none focus:border-gray-400"
                      >
                        {models.map((model) => (
                          <option key={model.value} value={model.value}>
                            {model.label}
                          </option>
                        ))}
                      </select>
                    </div>
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

            {(appState === "loading" || appState === "results" || appState === "error") && (
              <>
                <div className="flex items-center justify-between">
                  <h2 className="text-sm font-semibold">X-ray Viewer</h2>
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-gray-500">
                      {models.find((model) => model.value === (modelOutput?.modelUsed ?? selectedModel))?.label ?? "Local model"}
                    </span>
                    <Button variant="ghost" size="sm" onClick={handleReset} className="h-7 text-xs">
                      New
                    </Button>
                  </div>
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
                {appState === "error" && (
                  <Card className="border-red-200 bg-red-50">
                    <CardContent className="py-3">
                      <p className="text-xs font-medium text-red-900">Analysis failed</p>
                      <p className="text-xs text-red-800 mt-1 leading-relaxed">
                        {errorMessage}
                      </p>
                      <p className="text-xs text-red-700 mt-2">
                        Start the API on port 8001 or use the sample flow while the backend is offline.
                      </p>
                    </CardContent>
                  </Card>
                )}
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

                {appState === "results" && modelOutput && (
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
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Run History</CardTitle>
                <CardDescription className="text-xs">
                  Saved locally on this browser
                </CardDescription>
              </CardHeader>
              <CardContent className="pt-0">
                {runHistory.length === 0 ? (
                  <div className="h-56 rounded-md border border-dashed border-gray-200 bg-gray-50 px-4 py-6 text-center">
                    <p className="text-xs text-gray-600">No saved runs yet.</p>
                    <p className="text-xs text-gray-500 mt-1">
                      Completed analyses will appear here.
                    </p>
                  </div>
                ) : (
                  <div className="h-56 overflow-y-auto rounded-md border border-gray-200 bg-white">
                    <div className="divide-y divide-gray-100">
                      {runHistory.map((run) => (
                        <div
                          key={run.id}
                          className={`flex items-start gap-2 px-3 py-3 transition-colors ${
                            activeRunId === run.id ? "bg-gray-100" : "hover:bg-gray-50"
                          }`}
                        >
                          <button
                            type="button"
                            onClick={() => handleSelectRun(run)}
                            className="min-w-0 flex-1 text-left"
                          >
                            <p className="text-xs font-medium text-gray-900">
                              {formatRunTimestamp(run.createdAt)}
                            </p>
                            <p className="text-xs text-gray-600 mt-1">
                              Study ID: {run.studyId}
                            </p>
                            <p className="text-xs text-gray-500 mt-1 truncate">
                              {models.find((model) => model.value === (run.modelOutput.modelUsed ?? ""))?.label
                                ?? run.modelOutput.modelUsed
                                ?? "Saved run"}
                            </p>
                          </button>
                          <button
                            type="button"
                            aria-label={`Delete run ${run.studyId}`}
                            onClick={(event) => {
                              event.stopPropagation();
                              handleDeleteRun(run.id);
                            }}
                            className="mt-0.5 flex h-6 w-6 flex-shrink-0 items-center justify-center rounded text-gray-400 transition-colors hover:bg-gray-200 hover:text-gray-700"
                          >
                            <X className="h-3.5 w-3.5" />
                          </button>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>

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

            {(appState === "loading" || (appState === "results" && llmState === "loading")) && (
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm">AI Findings</CardTitle>
                  <CardDescription className="text-xs">
                    {appState === "loading" ? "Waiting for model output…" : "Generating explanation…"}
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-2">
                  <Skeleton className="h-3 w-full" />
                  <Skeleton className="h-3 w-full" />
                  <Skeleton className="h-3 w-11/12" />
                  <Skeleton className="h-3 w-full" />
                  <Skeleton className="h-3 w-4/5" />
                </CardContent>
              </Card>
            )}

            {appState === "results" && modelOutput && llmState !== "loading" && (
              <div className="space-y-3">
                {llmState === "error" && (
                  <Card className="border-amber-200 bg-amber-50">
                    <CardContent className="py-3">
                      <p className="text-xs font-medium text-amber-900">LLM explanation unavailable</p>
                      <p className="text-xs text-amber-800 mt-1 leading-relaxed">
                        {llmErrorMessage || "Classifier results loaded, but the explanation step failed."}
                      </p>
                    </CardContent>
                  </Card>
                )}
                <LLMCard
                  title="AI Findings"
                  description="Detailed synthesis generated from model outputs"
                >
                  <p className="text-xs leading-relaxed whitespace-pre-line">
                    {modelOutput.llm.summary || "The vision model results are available. The explanation step did not complete."}
                  </p>
                </LLMCard>
              </div>
            )}
          </div>
        </div>
      </div>
      <DisclaimerBanner text={footerDisclaimer} />
    </div>
  );
}

export default App;
