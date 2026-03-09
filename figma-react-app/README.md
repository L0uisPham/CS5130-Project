# Chest X-ray AI – React app with API connections

This folder contains a **single-file React app** that matches the Figma layout and **connects** to the backend.

## Connections

| UI element              | Data source              | Connection |
|-------------------------|--------------------------|------------|
| File drop / click       | User selects image       | `handleFileSelect(file)` → `POST ${API_URL}/analyze` with `FormData` |
| Image viewer            | Selected file            | `imageUrl` from `URL.createObjectURL(file)` |
| Model Predictions list   | API response             | `response.labels` → `modelOutput.labels` → `output.labels` |
| AI Summary              | API response             | `response.llm.summary` → `output.llm.summary` |
| Ranked Findings         | API response             | `response.llm.rankedFindings` |
| Differentials           | API response             | `response.llm.differentials` |
| Recommended Actions     | API response             | `response.llm.recommendedActions` |
| Disclaimer              | API response             | `response.llm.safetyNote` |

## Run

1. **Backend** (from repo root):  
   `python -m uvicorn api.app:app --reload --port 8000`  
   (Use `set API_MOCK=1` if you have no models/Ollama.)

2. **Frontend**: Copy `App.tsx` into your Vite/React app, or create a new Vite app and replace `src/App.tsx` with this file. Then:
   ```bash
   npm run dev
   ```
   Default API URL is `http://localhost:8000`. Override with `VITE_API_URL`.

## Figma

The **layout** is created in Figma by the plugin in `../figma-plugin-chest-xray/`. Frame names in Figma document the same bindings (e.g. “← response.labels”, “← response.llm.summary”).
