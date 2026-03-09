# Figma file and API connections

Two things give you “the Figma file” and “the connections”:

1. **Figma layout (the “file”)** – A **plugin** that creates the screen **inside Figma**.
2. **Connections** – **React app** + **API** that send the image to your models and show the result.

---

## 1. The “Figma file” (plugin that builds the screen)

**Location:** `figma-plugin-chest-xray/`

- **manifest.json** – Plugin manifest.
- **code.js** – Runs in Figma and creates one frame: header + 3 columns (Image | Model Predictions | LLM).
- **README.md** – How to install and run the plugin.

**Install in Figma**

1. In Figma: **Plugins → Development → Import plugin from manifest…**
2. Choose the **`figma-plugin-chest-xray`** folder (the one that contains `manifest.json`).
3. Run **Plugins → Development → Chest X-ray AI Analysis – Build Screen**.

You get one frame with the layout. **Layer names** describe the connections (e.g. “← response.labels”, “← response.llm.summary”), so the “Figma file” is the document you get after running the plugin.

---

## 2. The connections (React + API)

**Backend (this repo)**

- **`api/app.py`** – `POST /analyze` accepts an image, runs the pipeline (model + LLMollama), returns `{ labels, llm }`.
- Run: `python -m uvicorn api.app:app --reload --port 8000` from project root.

**Frontend (React that matches the Figma layout)**

- **`figma-react-app/App.tsx`** – Single-file React app that:
  - **Upload** → `handleFileSelect` → `fetch(POST /analyze)` with the image.
  - **State** → `modelOutput` holds `{ labels, llm }` from the API.
  - **UI** → Left column: image; middle: `output.labels`; right: `output.llm.summary`, `rankedFindings`, `differentials`, `recommendedActions`, `safetyNote`.

Use `figma-react-app/App.tsx` in your Vite/React app (or copy its logic). Set `VITE_API_URL` to your backend (default `http://localhost:8000`).

---

## Summary

| What you want        | Where it is |
|----------------------|-------------|
| Layout in Figma      | Run the plugin in `figma-plugin-chest-xray/` |
| Upload → API         | `figma-react-app/App.tsx` → `handleFileSelect` → `POST /analyze` |
| Show labels + LLM    | Same App: `modelOutput` / `output` from API response |
| API that runs models | `api/app.py` (see `FIGMA_BACKEND.md`) |

The “Figma file” is the frame the plugin creates. The “connections” are in the React app and the API.
