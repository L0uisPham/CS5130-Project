# Figma plugin: Chest X-ray AI Analysis – Build Screen

This plugin **creates the screen layout** in Figma and **names frames so you can see the API connections** in the layer panel.

## Install in Figma

1. In Figma: **Menu → Plugins → Development → Import plugin from manifest…**
2. Select the **folder** that contains `manifest.json` and `code.js` (this `figma-plugin-chest-xray` folder).
3. Run **Plugins → Development → Chest X-ray AI Analysis – Build Screen**.

## What it creates

- **One main frame** (1440×800): “Chest X-ray AI Analysis – Screen”
- **Header**: title, subtitle (“Connects to: POST /analyze”), and placeholder buttons (Export Report, New Analysis)
- **3 columns:**
  - **Left:** Upload area (labeled “file → POST /analyze”) and X-ray viewer
  - **Middle:** Model Predictions card (labeled “response.labels”)
  - **Right:** LLM cards, each frame named with the data it uses:
    - `response.llm.summary`
    - `response.llm.rankedFindings`
    - `response.llm.differentials`
    - `response.llm.recommendedActions`
    - `response.llm.safetyNote`

Use the **layer names** in Figma as the map to your React state and API: upload → `POST /analyze` → `response` → `modelOutput.labels` and `modelOutput.llm`.

## Connect the design to the app

The **React app** that matches this layout and performs the real connections (upload → API → state) is in the repo:

- **Backend:** `api/app.py` – `POST /analyze` returns `{ labels, llm }`.
- **Frontend:** See `FIGMA_APP_INTEGRATION.tsx` and `FIGMA_BACKEND.md` for the exact React code (fetch, state, using `modelOutput` / `output` instead of mock).

The Figma file is the **design**. The **connections** (upload → API → UI) live in the React and API code.
