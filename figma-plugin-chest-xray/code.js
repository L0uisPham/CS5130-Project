/**
 * Figma plugin: Chest X-ray AI Analysis – Build Screen
 * Run from Figma: Plugins → Development → Import plugin from manifest…
 * Point to the folder containing this file and manifest.json.
 *
 * Creates the screen layout that connects to:
 * - POST /analyze (image upload → labels + llm response)
 * - React state: modelOutput (labels, llm.summary, llm.rankedFindings, etc.)
 */

function createText(text, fontSize = 12, x = 0, y = 0) {
  const node = figma.createText();
  figma.loadFontAsync({ family: "Inter", style: "Regular" }).then(() => {
    node.characters = text;
    node.fontSize = fontSize;
    node.x = x;
    node.y = y;
  });
  return node;
}

function createLabel(parent, name, value) {
  const frame = figma.createFrame();
  frame.name = name;
  frame.resize(200, 24);
  frame.fills = [];
  const text = figma.createText();
  figma.loadFontAsync({ family: "Inter", style: "Regular" }).then(() => {
    text.characters = value || name;
    text.fontSize = 11;
    text.x = 0;
    text.y = 0;
    frame.appendChild(text);
    text.resize(200, 24);
  });
  parent.appendChild(frame);
  return frame;
}

function run() {
  const PAGE = figma.currentPage;
  const MARGIN = 24;
  const COL_GAP = 24;
  const HEADER_H = 80;
  const CONTENT_H = 720;
  const W = 1440;
  const H = HEADER_H + CONTENT_H + MARGIN * 2;

  // ─── Main frame ─────────────────────────────────────────────────────────
  const main = figma.createFrame();
  main.name = "Chest X-ray AI Analysis – Screen";
  main.resize(W, H);
  main.x = 0;
  main.y = 0;
  main.fills = [{ type: "SOLID", color: { r: 0.97, g: 0.97, b: 0.98 } }];
  main.layoutMode = "VERTICAL";
  main.primaryAxisSizingMode = "AUTO";
  main.counterAxisSizingMode = "FIXED";
  main.itemSpacing = 0;
  main.paddingLeft = main.paddingRight = main.paddingTop = main.paddingBottom = MARGIN;
  PAGE.appendChild(main);

  // ─── Header ─────────────────────────────────────────────────────────────
  const header = figma.createFrame();
  header.name = "Header [Export Report / New Analysis]";
  header.layoutMode = "HORIZONTAL";
  header.primaryAxisSizingMode = "FILL";
  header.counterAxisSizingMode = "AUTO";
  header.itemSpacing = 16;
  header.resize(W - MARGIN * 2, HEADER_H);
  header.fills = [{ type: "SOLID", color: { r: 1, g: 1, b: 1 } }];
  header.effects = [{ type: "DROP_SHADOW", color: { r: 0, g: 0, b: 0, a: 0.06 }, offset: { x: 0, y: 1 }, radius: 2 }];
  main.appendChild(header);

  const headerTitle = figma.createFrame();
  headerTitle.name = "Title block";
  headerTitle.layoutMode = "VERTICAL";
  headerTitle.primaryAxisSizingMode = "AUTO";
  headerTitle.counterAxisSizingMode = "AUTO";
  headerTitle.itemSpacing = 4;
  headerTitle.fills = [];
  header.appendChild(headerTitle);

  const titleText = figma.createText();
  figma.loadFontAsync({ family: "Inter", style: "Bold" }).then(() => {
    titleText.characters = "Chest X-ray AI Analysis";
    titleText.fontSize = 24;
    headerTitle.appendChild(titleText);
  });
  const subtitleText = figma.createText();
  figma.loadFontAsync({ family: "Inter", style: "Regular" }).then(() => {
    subtitleText.characters = "Vision Model + LLM Explanation Pipeline • Connects to: POST /analyze";
    subtitleText.fontSize = 12;
    subtitleText.fills = [{ type: "SOLID", color: { r: 0.45, g: 0.45, b: 0.5 } }];
    headerTitle.appendChild(subtitleText);
  });

  const headerSpacer = figma.createFrame();
  headerSpacer.name = "Spacer";
  headerSpacer.layoutMode = "HORIZONTAL";
  headerSpacer.primaryAxisSizingMode = "FILL";
  headerSpacer.resize(1, 1);
  headerSpacer.fills = [];
  header.appendChild(headerSpacer);

  const headerButtons = figma.createFrame();
  headerButtons.name = "Export Report | New Analysis";
  headerButtons.layoutMode = "HORIZONTAL";
  headerButtons.primaryAxisSizingMode = "AUTO";
  headerButtons.counterAxisSizingMode = "AUTO";
  headerButtons.itemSpacing = 12;
  headerButtons.fills = [];
  header.appendChild(headerButtons);

  const btnExport = figma.createFrame();
  btnExport.name = "Button: Export Report";
  btnExport.resize(140, 36);
  btnExport.fills = [];
  btnExport.strokes = [{ type: "SOLID", color: { r: 0.8, g: 0.8, b: 0.82 } }];
  btnExport.cornerRadius = 6;
  headerButtons.appendChild(btnExport);
  const btnNew = figma.createFrame();
  btnNew.name = "Button: New Analysis";
  btnNew.resize(120, 36);
  btnNew.fills = [];
  btnNew.strokes = [{ type: "SOLID", color: { r: 0.8, g: 0.8, b: 0.82 } }];
  btnNew.cornerRadius = 6;
  headerButtons.appendChild(btnNew);

  // ─── 3-column content ───────────────────────────────────────────────────
  const content = figma.createFrame();
  content.name = "3-Column Layout [Left: Image | Middle: Labels | Right: LLM]";
  content.layoutMode = "HORIZONTAL";
  content.primaryAxisSizingMode = "FILL";
  content.counterAxisSizingMode = "FIXED";
  content.counterAxisAlignItems = "MIN";
  content.itemSpacing = COL_GAP;
  content.resize(W - MARGIN * 2, CONTENT_H);
  content.fills = [];
  content.paddingTop = MARGIN;
  main.appendChild(content);

  const colW = (W - MARGIN * 2 - COL_GAP * 2) / 3;

  // Column 1: Image
  const col1 = figma.createFrame();
  col1.name = "Left: Image upload + X-ray Viewer ← file → POST /analyze";
  col1.layoutMode = "VERTICAL";
  col1.primaryAxisSizingMode = "FILL";
  col1.counterAxisSizingMode = "FIXED";
  col1.itemSpacing = 16;
  col1.resize(colW, CONTENT_H);
  col1.fills = [];
  content.appendChild(col1);

  const uploadArea = figma.createFrame();
  uploadArea.name = "UploadDropzone (file → handleFileSelect → fetch /analyze)";
  uploadArea.resize(colW, 120);
  uploadArea.fills = [{ type: "SOLID", color: { r: 0.95, g: 0.95, b: 0.96 } }];
  uploadArea.strokes = [{ type: "SOLID", color: { r: 0.85, g: 0.85, b: 0.88 } }];
  uploadArea.cornerRadius = 8;
  uploadArea.dashPattern = [8, 4];
  col1.appendChild(uploadArea);
  const uploadText = figma.createText();
  figma.loadFontAsync({ family: "Inter", style: "Regular" }).then(() => {
    uploadText.characters = "Drop X-ray or click to upload\n→ sends file to API";
    uploadText.fontSize = 12;
    uploadText.x = 16;
    uploadText.y = 44;
    uploadText.fills = [{ type: "SOLID", color: { r: 0.5, g: 0.5, b: 0.55 } }];
    uploadArea.appendChild(uploadText);
  });

  const viewerCard = figma.createFrame();
  viewerCard.name = "ImageViewerCard (imageUrl from state)";
  viewerCard.resize(colW, 400);
  viewerCard.fills = [{ type: "SOLID", color: { r: 1, g: 1, b: 1 } }];
  viewerCard.cornerRadius = 8;
  viewerCard.effects = [{ type: "DROP_SHADOW", color: { r: 0, g: 0, b: 0, a: 0.05 }, offset: { x: 0, y: 1 }, radius: 3 }];
  col1.appendChild(viewerCard);
  const viewerPlaceholder = figma.createText();
  figma.loadFontAsync({ family: "Inter", style: "Regular" }).then(() => {
    viewerPlaceholder.characters = "X-ray preview (imageUrl)";
    viewerPlaceholder.fontSize = 11;
    viewerPlaceholder.x = 16;
    viewerPlaceholder.y = 16;
    viewerPlaceholder.fills = [{ type: "SOLID", color: { r: 0.6, g: 0.6, b: 0.65 } }];
    viewerCard.appendChild(viewerPlaceholder);
  });

  // Column 2: Model predictions
  const col2 = figma.createFrame();
  col2.name = "Middle: Model Predictions ← response.labels";
  col2.layoutMode = "VERTICAL";
  col2.primaryAxisSizingMode = "FILL";
  col2.counterAxisSizingMode = "FIXED";
  col2.itemSpacing = 16;
  col2.resize(colW, CONTENT_H);
  col2.fills = [];
  content.appendChild(col2);

  const predictionsCard = figma.createFrame();
  predictionsCard.name = "Card: Model Predictions (Show Relevant / Show All)";
  predictionsCard.layoutMode = "VERTICAL";
  predictionsCard.primaryAxisSizingMode = "FILL";
  predictionsCard.counterAxisSizingMode = "FIXED";
  predictionsCard.itemSpacing = 12;
  predictionsCard.resize(colW, CONTENT_H - 20);
  predictionsCard.fills = [{ type: "SOLID", color: { r: 1, g: 1, b: 1 } }];
  predictionsCard.cornerRadius = 8;
  predictionsCard.effects = [{ type: "DROP_SHADOW", color: { r: 0, g: 0, b: 0, a: 0.05 }, offset: { x: 0, y: 1 }, radius: 3 }];
  predictionsCard.paddingLeft = predictionsCard.paddingRight = predictionsCard.paddingTop = predictionsCard.paddingBottom = 16;
  col2.appendChild(predictionsCard);

  const predHeader = figma.createText();
  figma.loadFontAsync({ family: "Inter", style: "Bold" }).then(() => {
    predHeader.characters = "Model Predictions";
    predHeader.fontSize = 16;
    predictionsCard.appendChild(predHeader);
  });
  const predDesc = figma.createText();
  figma.loadFontAsync({ family: "Inter", style: "Regular" }).then(() => {
    predDesc.characters = "CheXpert 14-label probabilities ← API response.labels";
    predDesc.fontSize = 12;
    predDesc.fills = [{ type: "SOLID", color: { r: 0.5, g: 0.5, b: 0.55 } }];
    predictionsCard.appendChild(predDesc);
  });
  const predList = figma.createFrame();
  predList.name = "ProbabilityRow list (label, probability, status)";
  predList.layoutMode = "VERTICAL";
  predList.primaryAxisSizingMode = "AUTO";
  predList.counterAxisSizingMode = "FILL";
  predList.itemSpacing = 8;
  predList.fills = [];
  predictionsCard.appendChild(predList);
  const predNote = figma.createText();
  figma.loadFontAsync({ family: "Inter", style: "Regular" }).then(() => {
    predNote.characters = "Each row: item.label, item.probability, item.status (present/uncertain/not-present)";
    predNote.fontSize = 10;
    predNote.fills = [{ type: "SOLID", color: { r: 0.55, g: 0.55, b: 0.6 } }];
    predList.appendChild(predNote);
  });

  // Column 3: LLM
  const col3 = figma.createFrame();
  col3.name = "Right: LLM Explanation ← response.llm";
  col3.layoutMode = "VERTICAL";
  col3.primaryAxisSizingMode = "FILL";
  col3.counterAxisSizingMode = "FIXED";
  col3.itemSpacing = 16;
  col3.resize(colW, CONTENT_H);
  col3.fills = [];
  content.appendChild(col3);

  const llmCards = [
    { name: "AI Findings Summary", bind: "response.llm.summary" },
    { name: "Ranked Findings", bind: "response.llm.rankedFindings" },
    { name: "Possible Differentials", bind: "response.llm.differentials" },
    { name: "Recommended Next Steps", bind: "response.llm.recommendedActions" },
    { name: "Disclaimer", bind: "response.llm.safetyNote" },
  ];

  llmCards.forEach((card, i) => {
    const frame = figma.createFrame();
    frame.name = `${card.name} ← ${card.bind}`;
    frame.layoutMode = "VERTICAL";
    frame.primaryAxisSizingMode = "AUTO";
    frame.counterAxisSizingMode = "FILL";
    frame.itemSpacing = 8;
    frame.resize(colW, 80);
    frame.fills = [{ type: "SOLID", color: { r: 1, g: 1, b: 1 } }];
    frame.cornerRadius = 8;
    frame.effects = [{ type: "DROP_SHADOW", color: { r: 0, g: 0, b: 0, a: 0.05 }, offset: { x: 0, y: 1 }, radius: 3 }];
    frame.paddingLeft = frame.paddingRight = frame.paddingTop = frame.paddingBottom = 12;
    col3.appendChild(frame);
    const title = figma.createText();
    figma.loadFontAsync({ family: "Inter", style: "Bold" }).then(() => {
      title.characters = card.name;
      title.fontSize = 14;
      frame.appendChild(title);
    });
    const bind = figma.createText();
    figma.loadFontAsync({ family: "Inter", style: "Regular" }).then(() => {
      bind.characters = card.bind;
      bind.fontSize = 10;
      bind.fills = [{ type: "SOLID", color: { r: 0.4, g: 0.5, b: 0.7 } }];
      frame.appendChild(bind);
    });
  });

  figma.currentPage.selection = [main];
  figma.viewport.scrollAndZoomIntoView([main]);
  figma.closePlugin("Chest X-ray AI screen created. Connections: file → POST /analyze → response.labels + response.llm");
}

run();
