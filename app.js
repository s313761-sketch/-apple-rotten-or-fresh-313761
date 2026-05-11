// ===============================
// YOLOv8 ONNX Web Object Detection
// GitHub Pages 可用版本
// ===============================

const MODEL_PATH = "models/best.onnx";

// 若 ONNX 是用 imgsz=960 匯出，請改成 960。
const INPUT_SIZE = 640;

// 三種類別價值，必須對應 Roboflow / YOLO data.yaml 的類別順序。
const CLASS_VALUES = {
  0: 100,
  1: 200,
  2: 300
};

const CLASS_COLORS = ["#22c55e", "#f97316", "#38bdf8", "#e879f9", "#facc15", "#fb7185"];

let session = null;
let labels = [];
let uploadedImage = null;
let cameraStream = null;
let cameraRunning = false;
let animationId = null;
let lastCameraDetectTime = 0;
const CAMERA_DETECT_INTERVAL = 160;

const imageInput = document.getElementById("imageInput");
const detectImageBtn = document.getElementById("detectImageBtn");
const startCameraBtn = document.getElementById("startCameraBtn");
const stopCameraBtn = document.getElementById("stopCameraBtn");
const confSlider = document.getElementById("confSlider");
const iouSlider = document.getElementById("iouSlider");
const confValue = document.getElementById("confValue");
const iouValue = document.getElementById("iouValue");
const statusEl = document.getElementById("status");
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const statsBody = document.getElementById("statsBody");
const totalCountEl = document.getElementById("totalCount");
const totalValueEl = document.getElementById("totalValue");

init();

async function init() {
  setButtonsDisabled(true);
  updateStatus("正在載入 labels.json 與 ONNX 模型...");

  try {
    labels = await loadLabels();
    initStatsTable();

    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
    session = await ort.InferenceSession.create(MODEL_PATH, { executionProviders: ["wasm"] });

    updateStatus("模型載入完成。請上傳圖片或啟動手機後鏡頭。");
    setButtonsDisabled(false);
  } catch (error) {
    console.error(error);
    updateStatus("模型載入失敗。請確認 models/best.onnx 是否存在，且 labels.json 格式正確。");
  }
}

async function loadLabels() {
  const response = await fetch("labels.json");
  if (!response.ok) throw new Error("無法讀取 labels.json");
  return await response.json();
}

function setButtonsDisabled(disabled) {
  detectImageBtn.disabled = disabled;
  startCameraBtn.disabled = disabled;
  stopCameraBtn.disabled = disabled;
}

function updateStatus(message) {
  statusEl.textContent = message;
}

confSlider.addEventListener("input", () => {
  confValue.textContent = Number(confSlider.value).toFixed(2);
});

iouSlider.addEventListener("input", () => {
  iouValue.textContent = Number(iouSlider.value).toFixed(2);
});

imageInput.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;

  stopCamera();
  uploadedImage = await loadImageFromFile(file);
  drawSourceToCanvas(uploadedImage);
  updateStatus("圖片已載入，請按「辨識上傳圖片」。");
});

detectImageBtn.addEventListener("click", async () => {
  if (!uploadedImage) {
    updateStatus("請先選擇一張圖片。");
    return;
  }
  await detectStillImage(uploadedImage);
});

startCameraBtn.addEventListener("click", async () => {
  await startCamera();
});

stopCameraBtn.addEventListener("click", () => {
  stopCamera();
  updateStatus("鏡頭已停止。");
});

function loadImageFromFile(file) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = URL.createObjectURL(file);
  });
}

function drawSourceToCanvas(source) {
  const maxWidth = 1280;
  const scale = Math.min(1, maxWidth / source.width);
  canvas.width = Math.round(source.width * scale);
  canvas.height = Math.round(source.height * scale);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(source, 0, 0, canvas.width, canvas.height);
}

async function detectStillImage(image) {
  if (!session) return;
  updateStatus("圖片辨識中...");

  drawSourceToCanvas(image);
  const detections = await runDetection(canvas);

  drawSourceToCanvas(image);
  drawDetections(ctx, detections);
  drawStatsPanel(ctx, detections);
  updateStatsTable(detections);

  updateStatus(`辨識完成，共偵測到 ${detections.length} 個物件。`);
}

async function startCamera() {
  if (!session) return;
  stopCamera();

  try {
    updateStatus("正在啟動手機後鏡頭...");
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: 1280 },
        height: { ideal: 720 }
      },
      audio: false
    });

    video.srcObject = cameraStream;
    await video.play();
    cameraRunning = true;
    uploadedImage = null;
    updateStatus("鏡頭已啟動，正在即時辨識...");
    detectCameraLoop();
  } catch (error) {
    console.error(error);
    updateStatus("無法啟動鏡頭。請確認瀏覽器權限，並使用 HTTPS 或 GitHub Pages 開啟。");
  }
}

function stopCamera() {
  cameraRunning = false;
  if (animationId) {
    cancelAnimationFrame(animationId);
    animationId = null;
  }
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = null;
  }
  video.srcObject = null;
}

async function detectCameraLoop(timestamp = 0) {
  if (!cameraRunning) return;

  if (video.videoWidth > 0 && video.videoHeight > 0) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    if (timestamp - lastCameraDetectTime > CAMERA_DETECT_INTERVAL) {
      lastCameraDetectTime = timestamp;
      try {
        const detections = await runDetection(canvas);
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        drawDetections(ctx, detections);
        drawStatsPanel(ctx, detections);
        updateStatsTable(detections);
      } catch (error) {
        console.error(error);
      }
    }
  }

  animationId = requestAnimationFrame(detectCameraLoop);
}

async function runDetection(sourceCanvas) {
  const confThreshold = Number(confSlider.value);
  const iouThreshold = Number(iouSlider.value);

  const { inputTensor, ratio, xPad, yPad } = preprocess(sourceCanvas);
  const feeds = {};
  feeds[session.inputNames[0]] = inputTensor;

  const output = await session.run(feeds);
  const outputTensor = output[session.outputNames[0]];

  let detections = parseYOLOv8Output(outputTensor, confThreshold, sourceCanvas.width, sourceCanvas.height, ratio, xPad, yPad);
  detections = nonMaxSuppression(detections, iouThreshold);
  return detections;
}

function preprocess(sourceCanvas) {
  const srcW = sourceCanvas.width;
  const srcH = sourceCanvas.height;
  const ratio = Math.min(INPUT_SIZE / srcW, INPUT_SIZE / srcH);
  const newW = Math.round(srcW * ratio);
  const newH = Math.round(srcH * ratio);
  const xPad = Math.floor((INPUT_SIZE - newW) / 2);
  const yPad = Math.floor((INPUT_SIZE - newH) / 2);

  const tmpCanvas = document.createElement("canvas");
  tmpCanvas.width = INPUT_SIZE;
  tmpCanvas.height = INPUT_SIZE;
  const tmpCtx = tmpCanvas.getContext("2d");

  tmpCtx.fillStyle = "rgb(114,114,114)";
  tmpCtx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
  tmpCtx.drawImage(sourceCanvas, 0, 0, srcW, srcH, xPad, yPad, newW, newH);

  const imageData = tmpCtx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
  const red = new Float32Array(INPUT_SIZE * INPUT_SIZE);
  const green = new Float32Array(INPUT_SIZE * INPUT_SIZE);
  const blue = new Float32Array(INPUT_SIZE * INPUT_SIZE);

  let p = 0;
  for (let i = 0; i < imageData.length; i += 4) {
    red[p] = imageData[i] / 255;
    green[p] = imageData[i + 1] / 255;
    blue[p] = imageData[i + 2] / 255;
    p++;
  }

  const input = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
  input.set(red, 0);
  input.set(green, INPUT_SIZE * INPUT_SIZE);
  input.set(blue, 2 * INPUT_SIZE * INPUT_SIZE);

  const inputTensor = new ort.Tensor("float32", input, [1, 3, INPUT_SIZE, INPUT_SIZE]);
  return { inputTensor, ratio, xPad, yPad };
}

function parseYOLOv8Output(outputTensor, confThreshold, originalWidth, originalHeight, ratio, xPad, yPad) {
  const data = outputTensor.data;
  const dims = outputTensor.dims;
  const numClasses = labels.length;
  let numBoxes;
  let boxLength;
  let isTransposed = false;

  if (dims.length !== 3) throw new Error("不支援的模型輸出維度：" + dims.join(","));

  if (dims[1] === 4 + numClasses) {
    boxLength = dims[1];
    numBoxes = dims[2];
    isTransposed = true;
  } else {
    numBoxes = dims[1];
    boxLength = dims[2];
    isTransposed = false;
  }

  const detections = [];

  for (let i = 0; i < numBoxes; i++) {
    const getValue = (j) => isTransposed ? data[j * numBoxes + i] : data[i * boxLength + j];

    const cx = getValue(0);
    const cy = getValue(1);
    const w = getValue(2);
    const h = getValue(3);

    let bestScore = -Infinity;
    let classId = -1;

    for (let c = 0; c < numClasses; c++) {
      const score = getValue(4 + c);
      if (score > bestScore) {
        bestScore = score;
        classId = c;
      }
    }

    if (bestScore < confThreshold) continue;

    let x1 = cx - w / 2;
    let y1 = cy - h / 2;
    let x2 = cx + w / 2;
    let y2 = cy + h / 2;

    x1 = (x1 - xPad) / ratio;
    y1 = (y1 - yPad) / ratio;
    x2 = (x2 - xPad) / ratio;
    y2 = (y2 - yPad) / ratio;

    x1 = clamp(x1, 0, originalWidth);
    y1 = clamp(y1, 0, originalHeight);
    x2 = clamp(x2, 0, originalWidth);
    y2 = clamp(y2, 0, originalHeight);

    const boxW = x2 - x1;
    const boxH = y2 - y1;
    if (boxW <= 1 || boxH <= 1) continue;

    detections.push({ x: x1, y: y1, w: boxW, h: boxH, score: bestScore, classId });
  }
  return detections;
}

function nonMaxSuppression(detections, iouThreshold) {
  const results = [];
  const grouped = {};

  for (const det of detections) {
    if (!grouped[det.classId]) grouped[det.classId] = [];
    grouped[det.classId].push(det);
  }

  for (const classId of Object.keys(grouped)) {
    const boxes = grouped[classId].sort((a, b) => b.score - a.score);
    while (boxes.length > 0) {
      const chosen = boxes.shift();
      results.push(chosen);
      for (let i = boxes.length - 1; i >= 0; i--) {
        if (iou(chosen, boxes[i]) > iouThreshold) boxes.splice(i, 1);
      }
    }
  }
  return results.sort((a, b) => b.score - a.score);
}

function iou(a, b) {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w);
  const y2 = Math.min(a.y + a.h, b.y + b.h);
  const interW = Math.max(0, x2 - x1);
  const interH = Math.max(0, y2 - y1);
  const interArea = interW * interH;
  const areaA = a.w * a.h;
  const areaB = b.w * b.h;
  return interArea / (areaA + areaB - interArea + 1e-6);
}

function drawDetections(ctx, detections) {
  ctx.save();
  for (const det of detections) {
    const color = CLASS_COLORS[det.classId % CLASS_COLORS.length];
    const label = labels[det.classId] ?? `類別${det.classId}`;
    const text = `${label} ${(det.score * 100).toFixed(1)}%`;

    ctx.lineWidth = Math.max(2, Math.round(canvas.width / 400));
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.strokeRect(det.x, det.y, det.w, det.h);

    ctx.font = `${Math.max(16, Math.round(canvas.width / 55))}px Arial, Microsoft JhengHei`;
    const textMetrics = ctx.measureText(text);
    const textHeight = Math.max(22, Math.round(canvas.width / 42));
    const textWidth = textMetrics.width + 12;
    const textX = det.x;
    const textY = Math.max(0, det.y - textHeight);

    ctx.fillRect(textX, textY, textWidth, textHeight);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(text, textX + 6, textY + textHeight - 6);
  }
  ctx.restore();
}

function calculateStats(detections) {
  const stats = {};
  let totalCount = 0;
  let totalValue = 0;

  labels.forEach((label, index) => {
    const unitValue = Number(CLASS_VALUES[index] ?? 0);
    stats[index] = { classId: index, name: label, count: 0, unitValue, totalValue: 0 };
  });

  detections.forEach((det) => {
    const classId = Number(det.classId);
    const unitValue = Number(CLASS_VALUES[classId] ?? 0);
    if (!stats[classId]) {
      stats[classId] = { classId, name: labels[classId] ?? `類別${classId}`, count: 0, unitValue, totalValue: 0 };
    }
    stats[classId].count += 1;
    stats[classId].totalValue += unitValue;
    totalCount += 1;
    totalValue += unitValue;
  });

  return { stats, totalCount, totalValue };
}

function updateStatsTable(detections) {
  const { stats, totalCount, totalValue } = calculateStats(detections);
  totalCountEl.textContent = String(totalCount);
  totalValueEl.textContent = `${totalValue.toLocaleString()} 元`;
  statsBody.innerHTML = "";

  Object.values(stats).forEach((item) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${item.name}</td><td>${item.count}</td><td>${item.unitValue.toLocaleString()} 元</td><td>${item.totalValue.toLocaleString()} 元</td>`;
    statsBody.appendChild(tr);
  });
}

function initStatsTable() {
  updateStatsTable([]);
}

function drawStatsPanel(ctx, detections) {
  const { stats, totalCount, totalValue } = calculateStats(detections);
  const lines = [`總數：${totalCount}`, `總價值：${totalValue.toLocaleString()} 元`];

  Object.values(stats).forEach((item) => {
    lines.push(`${item.name}：${item.count} 個 × ${item.unitValue.toLocaleString()} 元 = ${item.totalValue.toLocaleString()} 元`);
  });

  ctx.save();
  const fontSize = Math.max(16, Math.round(canvas.width / 60));
  const lineHeight = Math.round(fontSize * 1.5);
  const padding = 10;
  ctx.font = `${fontSize}px Arial, Microsoft JhengHei`;
  const maxTextWidth = Math.max(...lines.map((line) => ctx.measureText(line).width));
  const panelWidth = Math.min(canvas.width - 20, maxTextWidth + padding * 2);
  const panelHeight = padding * 2 + lines.length * lineHeight;

  ctx.globalAlpha = 0.78;
  ctx.fillStyle = "#000000";
  ctx.fillRect(10, 10, panelWidth, panelHeight);

  ctx.globalAlpha = 1;
  ctx.fillStyle = "#ffffff";
  ctx.textBaseline = "top";
  lines.forEach((line, index) => ctx.fillText(line, 10 + padding, 10 + padding + index * lineHeight));
  ctx.restore();
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}
