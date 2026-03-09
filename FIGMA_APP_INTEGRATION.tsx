/**
 * Use this to connect your Figma/React app to the GitHub backend.
 *
 * 1. In your App.tsx, add state for the API result and an API base URL.
 * 2. Replace handleFileSelect so it POSTs the file to /analyze and sets the result.
 * 3. Use modelOutput instead of MOCK_OUTPUT when rendering results.
 *
 * Copy the snippets below into your existing App.tsx, or use this as reference.
 */

// ---------- 1) Add these near your other useState calls ----------

const API_URL = import.meta.env?.VITE_API_URL ?? "http://localhost:8000";
const [modelOutput, setModelOutput] = useState<ModelOutput | null>(null);
const [error, setError] = useState<string | null>(null);

// ---------- 2) Replace handleFileSelect with this (async, calls API) ----------

const handleFileSelect = async (file: File) => {
  const url = URL.createObjectURL(file);
  setImageUrl(url);
  setAppState("loading");
  setError(null);

  try {
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch(`${API_URL}/analyze`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(errText || `Analysis failed: ${res.status}`);
    }

    const data: ModelOutput = await res.json();
    setModelOutput(data);
    setAppState("results");
  } catch (e) {
    console.error(e);
    setError(e instanceof Error ? e.message : "Analysis failed");
    setAppState("empty");
    setImageUrl("");
  }
};

// ---------- 3) In handleReset, clear model output ----------

const handleReset = () => {
  setAppState("empty");
  setImageUrl("");
  setFilterMode("all");
  setModelOutput(null);
  setError(null);
};

// ---------- 4) Use modelOutput instead of MOCK_OUTPUT when showing results ----------

// Replace:
//   const filteredLabels = filterMode === "relevant"
//     ? MOCK_OUTPUT.labels.filter(...)
//     : MOCK_OUTPUT.labels;
// With:
const output = modelOutput ?? MOCK_OUTPUT; // fallback to mock if no API result yet
const filteredLabels =
  filterMode === "relevant"
    ? output.labels.filter((l) => l.status !== "not-present")
    : output.labels;

// Then everywhere you use MOCK_OUTPUT.labels or MOCK_OUTPUT.llm in the "results" view,
// use `output` instead (e.g. output.llm.summary, output.llm.rankedFindings, etc.).

// ---------- 5) Optional: show error in the UI ----------

{error && (
  <div className="bg-red-50 border border-red-200 text-red-800 px-4 py-2 rounded mb-4">
    {error}
  </div>
)}
