# Connect your Figma/React app to the GitHub models

This repo exposes an **API** that runs the vision model + LLMollama pipeline and returns JSON in the exact shape your React app expects (`ModelOutput`).

---

## 1. Start the backend (from project root)

```cmd
cd c:\Users\Ray\Desktop\CS5130\CS5130-Project\CS5130-Project
pip install fastapi uvicorn python-multipart
python -m uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

- **With real models + Ollama:** Ensure `models/resnet34_chexpert.pt` (or sk weights) and Ollama are available.
- **Frontend-only (no models, no Ollama):**  
  `set API_MOCK=1`  
  then run uvicorn. Every **POST /analyze** will return static mock data (no model or LLM run). Good for wiring the React app.
- **With models but no Ollama:**  
  `set LLMOLLAMA_DRY_RUN=1`  
  then run uvicorn. Real model runs; LLM output is mock.

Check: open **http://localhost:8000/health** — you should see `{"status":"ok"}`.

---

## 2. Point the React app at the API

In your React app, replace the **mock flow** (e.g. `setTimeout` that sets `MOCK_OUTPUT`) with a real **POST** to the backend.

### Option A: Environment variable for API URL

Create or edit `.env` in the React app:

```env
VITE_API_URL=http://localhost:8000
```

(or `NEXT_PUBLIC_API_URL=http://localhost:8000` for Next.js, or `REACT_APP_API_URL` for CRA.)

### Option B: Call the API when the user uploads a file

Replace the simulated “model processing” with a `fetch` to your backend. Example:

```ts
const API_URL = import.meta.env?.VITE_API_URL ?? "http://localhost:8000";

const handleFileSelect = async (file: File) => {
  const url = URL.createObjectURL(file);
  setImageUrl(url);
  setAppState("loading");

  try {
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch(`${API_URL}/analyze`, {
      method: "POST",
      body: formData,
      // Do not set Content-Type; browser sets it with boundary for FormData
    });

    if (!res.ok) {
      const err = await res.text();
      throw new Error(err || `Analysis failed: ${res.status}`);
    }

    const data: ModelOutput = await res.json();
    setModelOutput(data);  // Store in state (see below)
    setAppState("results");
  } catch (e) {
    console.error(e);
    setAppState("empty");
    setImageUrl("");
    // Optionally set error state and show a toast
  }
};
```

### Option C: State for the API result

Add state for the pipeline result and use it instead of `MOCK_OUTPUT`:

```ts
const [modelOutput, setModelOutput] = useState<ModelOutput | null>(null);
```

After a successful `fetch`, call `setModelOutput(data)`. In the JSX, use `modelOutput` instead of `MOCK_OUTPUT`:

- `modelOutput?.labels` / `modelOutput?.llm` (with optional chaining and null checks).
- When `appState === "results"` and `modelOutput` is set, render `modelOutput.llm.summary`, `modelOutput.llm.rankedFindings`, etc., exactly like you do with `MOCK_OUTPUT` now.

---

## 3. Response shape (same as your `ModelOutput`)

The **POST /analyze** response matches your TypeScript types:

- **labels:** `Array<{ label: string; probability: number; status: "present" | "uncertain" | "not-present" }>`
- **llm.summary:** string  
- **llm.rankedFindings:** `Array<{ label; status; probability; rationale }>`  
- **llm.differentials:** `Array<{ condition; confidence: "low"|"medium"|"high"; reason }>`  
- **llm.recommendedActions:** `Array<{ action; urgency: "routine"|"soon"|"urgent" }>`  
- **llm.safetyNote:** string  

So you can type the response as `ModelOutput` and use it directly in your existing UI.

---

## 4. Optional: Export report

If you want “Export Report” to download the **full prose report** (from `write_personalized_report`), the backend can return it in the response and the frontend can offer it as a download. Right now the API returns only the **structured** `ModelOutput` (labels + LLM summary/findings/differentials/actions/safetyNote). To add the long-form report, we’d add a field like `report: string` to the API response and use that for the export. If you want that, say so and we can add it.

---

## 5. CORS

The API allows origins:

- `http://localhost:3000`
- `http://localhost:5173`
- `http://127.0.0.1:3000`
- `http://127.0.0.1:5173`

Run your React app on one of these (e.g. Vite default 5173) and the browser will allow requests to **http://localhost:8000**.

---

## Quick checklist

1. Backend: `python -m uvicorn api.app:app --reload --host 0.0.0.0 --port 8000` (from project root).
2. React: Replace mock with `fetch(`${API_URL}/analyze`, { method: "POST", body: formData })` and set state from `res.json()`.
3. UI: Use the state object (e.g. `modelOutput`) instead of `MOCK_OUTPUT` when `appState === "results"`.

After that, your Figma UI is connected to the GitHub models and LLMollama.
