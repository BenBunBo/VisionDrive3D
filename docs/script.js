function fmt(value, digits = 3) {
  const num = Number(value);
  if (Number.isNaN(num) || !Number.isFinite(num)) return "-";
  return num.toFixed(digits);
}

function getRank(value, allValues, isMinBetter = false) {
  const sorted = [...new Set(allValues)].sort((a, b) => isMinBetter ? a - b : b - a);
  const index = sorted.findIndex((v) => Math.abs(v - value) < 1e-9);
  return index === -1 ? null : index + 1;
}

function renderQualitative(meta) {
  const grid = document.getElementById("qualitative-grid");
  grid.innerHTML = "";

  const scenes = ((meta && meta.qualitative_scenes) || []).slice(0, 3);
  if (!scenes.length) {
    const p = document.createElement("p");
    p.className = "notice";
    p.textContent = "No qualitative metadata found yet. Run evaluation first.";
    grid.appendChild(p);
    return;
  }

  scenes.forEach((stem) => {
    const row = document.createElement("div");
    row.className = "qual-row";
    const heading = document.createElement("h3");
    heading.className = "qual-scene-title";
    heading.textContent = stem;
    row.appendChild(heading);

    const comparisons = [
      {
        label: "Detection",
        baseline: `assets/images/qualitative/detection_baseline_${stem}.png`,
        finetuned: `assets/images/qualitative/detection_finetuned_${stem}.png`,
      },
      {
        label: "Segmentation",
        baseline: `assets/images/qualitative/semantic_baseline_${stem}.png`,
        finetuned: `assets/images/qualitative/semantic_finetuned_${stem}.png`,
      },
    ];

    comparisons.forEach((comparison) => {
      const card = document.createElement("section");
      card.className = "qual-card";

      const title = document.createElement("h4");
      title.textContent = comparison.label;
      card.appendChild(title);

      const pair = document.createElement("div");
      pair.className = "qual-preview-pair";
      [
        { label: "Baseline", file: comparison.baseline },
        { label: "Fine-tuned", file: comparison.finetuned },
      ].forEach((result) => {
        const item = document.createElement("div");
        item.className = "qual-result";

        const resultLabel = document.createElement("p");
        resultLabel.textContent = result.label;
        item.appendChild(resultLabel);

        const button = document.createElement("button");
        button.className = "qual-zoom";
        button.type = "button";
        button.dataset.pairBaseline = comparison.baseline;
        button.dataset.pairFinetuned = comparison.finetuned;
        button.dataset.pairTitle = `${comparison.label} — ${stem}`;
        button.setAttribute("aria-label", `Compare ${comparison.label} baseline and fine-tuned results for ${stem}`);

        const image = document.createElement("img");
        image.src = result.file;
        image.alt = `${comparison.label} ${result.label.toLowerCase()} result`;
        image.loading = "lazy";
        button.appendChild(image);
        item.appendChild(button);
        pair.appendChild(item);
      });
      card.appendChild(pair);
      row.appendChild(card);
    });

    grid.appendChild(row);
  });
}

async function loadResults() {
  try {
    const response = await fetch("assets/data/results.json?v=8");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();

    renderQualitative(data.meta || {});
  } catch (err) {
    console.error(err);
  }
}

function renderZeroShotTable(data) {
  const body = document.getElementById("zeroshot-body");
  body.innerHTML = "";

  const displayNames = {
    yolo26n: "YOLO26n",
    yolo26s: "YOLO26s",
    yolo26m: "YOLO26m",
    yolo26l: "YOLO26l",
    yolo26x: "YOLO26x",
    "rtdetrv2-s": "RT-DETRv2-S (R18)",
    fasterrcnn: "Faster R-CNN (R50-FPN)",
  };
  const models = (data && data.models) || {};
  const rows = Object.keys(displayNames)
    .filter((key) => models[key])
    .map((key) => ({
      model: displayNames[key],
      resolution: models[key].input_resolution || "-",
      map50: Number(models[key].mAP50 || 0),
      map95: Number(models[key].mAP50_95 || 0),
      fps: Number(models[key].FPS || 0),
      paramsM: Number(models[key].params || 0) / 1e6,
      flopsG: Number(models[key].FLOPs || 0) / 1e9,
    }));

  if (!rows.length) {
    body.innerHTML = '<tr><td colspan="7" class="notice">benchmark_zeroshot.json not found.</td></tr>';
    return;
  }

  const allMap50 = rows.map((r) => r.map50);
  const allMap95 = rows.map((r) => r.map95);
  const allFps = rows.map((r) => r.fps);
  const allParams = rows.map((r) => r.paramsM);
  const allFlops = rows.map((r) => r.flopsG);

  const getClass = (rank) => {
    if (rank === 1) return "best";
    if (rank === 2) return "second-best";
    if (rank === 3) return "third-best";
    return "";
  };

  rows.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.model}</td>
      <td>${row.resolution}</td>
      <td class="${getClass(getRank(row.map50, allMap50, false))}">${fmt(row.map50, 3)}</td>
      <td class="${getClass(getRank(row.map95, allMap95, false))}">${fmt(row.map95, 3)}</td>
      <td class="${getClass(getRank(row.fps, allFps, false))}">${fmt(row.fps, 1)}</td>
      <td class="${getClass(getRank(row.paramsM, allParams, true))}">${fmt(row.paramsM, 1)}</td>
      <td class="${getClass(getRank(row.flopsG, allFlops, true))}">${fmt(row.flopsG, 1)}</td>
    `;
    body.appendChild(tr);
  });
}

async function loadZeroShot() {
  try {
    const response = await fetch("assets/data/benchmark_zeroshot.json?v=6");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    renderZeroShotTable(data);
  } catch (err) {
    console.error(err);
    document.getElementById("zeroshot-body").innerHTML =
      '<tr><td colspan="7" class="notice">benchmark_zeroshot.json not found.</td></tr>';
  }
}

async function loadSemanticSegmentation() {
  const body = document.getElementById("semantic-segmentation-body");
  if (!body) return;
  const displayNames = {
    "yolo26n-sem": "YOLO26n-sem",
    "yolo26s-sem": "YOLO26s-sem",
    "yolo26m-sem": "YOLO26m-sem",
    "yolo26l-sem": "YOLO26l-sem",
    "yolo26x-sem": "YOLO26x-sem",
  };
  try {
    const response = await fetch("assets/data/benchmark_semantic_segmentation.json?v=1");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    body.replaceChildren();
    const numeric = (value) => value === null || value === undefined ? NaN : Number(value);
    const rows = Object.entries(displayNames).map(([key, name]) => {
      const model = (data.models || {})[key] || {};
      return {
        name,
        status: model.status,
        resolution: model.input_resolution || "-",
        carIou: numeric(model.car_iou),
        carDice: numeric(model.car_dice),
        miou: numeric(model.miou),
        fps: numeric(model.FPS || (model.time_ms ? 1000 / model.time_ms : null)),
        params: numeric(model.params) / 1e6,
        flops: numeric(model.FLOPs) / 1e9,
      };
    });
    const rankClass = (value, values, isMinBetter = false) => {
      if (!Number.isFinite(value)) return "";
      const rank = getRank(value, values, isMinBetter);
      return rank === 1 ? "best" : rank === 2 ? "second-best" : rank === 3 ? "third-best" : "";
    };
    const completeRows = rows.filter((row) => row.status === "complete");
    const validValues = (key) => completeRows.map((row) => row[key]).filter(Number.isFinite);
    const carIouValues = validValues("carIou");
    const carDiceValues = validValues("carDice");
    const miouValues = validValues("miou");
    const fpsValues = validValues("fps");
    const paramValues = validValues("params");
    const flopValues = validValues("flops");
    rows.forEach((item) => {
      const tr = document.createElement("tr");
      if (item.status !== "complete") {
        tr.innerHTML = `<td>${item.name}</td><td>${item.resolution}</td><td colspan="6">${item.status === "error" ? "Error" : "Pending GPU benchmark"}</td>`;
      } else {
        tr.innerHTML = `
          <td>${item.name}</td>
          <td>${item.resolution}</td>
          <td class="${rankClass(item.carIou, carIouValues)}">${fmt(item.carIou, 3)}</td>
          <td class="${rankClass(item.carDice, carDiceValues)}">${fmt(item.carDice, 3)}</td>
          <td class="${rankClass(item.miou, miouValues)}">${fmt(item.miou, 3)}</td>
          <td class="${rankClass(item.fps, fpsValues)}">${fmt(item.fps, 1)}</td>
          <td class="${rankClass(item.params, paramValues, true)}">${fmt(item.params, 1)}</td>
          <td class="${rankClass(item.flops, flopValues, true)}">${fmt(item.flops, 1)}</td>`;
      }
      body.appendChild(tr);
    });
  } catch (error) {
    console.error(error);
    body.innerHTML = '<tr><td colspan="8" class="notice">Semantic segmentation benchmark data unavailable.</td></tr>';
  }
}

async function loadYolo26Comparison() {
  const body = document.getElementById("yolo26-comparison-body");
  if (!body) return;
  const versions = [
    ["yolo26n", "baseline", "YOLO26n", "Detection", "COCO baseline"],
    ["yolo26n", "finetuned", "YOLO26n", "Detection", "Fine-tuned"],
    ["yolo26n-sem", "baseline", "YOLO26n-sem", "Semantic segmentation", "Cityscapes baseline"],
    ["yolo26n-sem", "finetuned", "YOLO26n-sem", "Semantic segmentation", "Fine-tuned"],
  ];
  try {
    const response = await fetch("assets/data/benchmark_yolo26_finetune.json?v=9");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    body.replaceChildren();
    versions.forEach(([key, version, name, task, versionName]) => {
      const metrics = (((data || {}).models || {})[key] || {})[version];
      const tr = document.createElement("tr");
      const values = metrics ? [
        metrics.mAPbox_50, metrics.mAPbox_50_95,
        metrics.car_iou, metrics.car_dice, metrics.miou,
      ] : [];
      tr.innerHTML = `<td>${name}</td><td>${task}</td><td>${versionName}</td>${values.map((value) =>
        `<td>${value === null || value === undefined ? "—" : fmt(value, 3)}</td>`).join("")}`;
      body.appendChild(tr);
    });
  } catch (error) {
    console.error(error);
    body.innerHTML = '<tr><td colspan="8" class="notice">YOLO26 fine-tuning comparison data unavailable.</td></tr>';
  }
}

loadResults();
loadZeroShot();
loadSemanticSegmentation();
loadYolo26Comparison();

// Teaser lightbox
(function () {
  const lightbox = document.getElementById("lightbox");
  const lbImg = document.getElementById("lightbox-img");
  const lbCaption = document.getElementById("lightbox-caption");
  const lbPair = document.getElementById("lightbox-pair");
  const lbBaselineImg = document.getElementById("lightbox-baseline-img");
  const lbFinetunedImg = document.getElementById("lightbox-finetuned-img");
  const lbClose = document.getElementById("lightbox-close");
  if (!lightbox || !lbImg || !lbCaption || !lbPair || !lbBaselineImg || !lbFinetunedImg || !lbClose) return;

  function open(src, caption) {
    lbPair.hidden = true;
    lbImg.hidden = false;
    lbImg.src = src;
    lbImg.alt = caption;
    lbCaption.textContent = caption;
    lightbox.hidden = false;
    document.body.style.overflow = "hidden";
  }

  function openPair(baseline, finetuned, caption) {
    lbImg.hidden = true;
    lbPair.hidden = false;
    lbBaselineImg.src = baseline;
    lbFinetunedImg.src = finetuned;
    lbCaption.textContent = caption;
    lightbox.hidden = false;
    document.body.style.overflow = "hidden";
  }

  function close() {
    lightbox.hidden = true;
    document.body.style.overflow = "";
  }

  document.addEventListener("click", (event) => {
    const btn = event.target.closest(".teaser-zoom, .qual-zoom");
    if (!btn) return;
    if (btn.dataset.pairBaseline && btn.dataset.pairFinetuned) {
      openPair(btn.dataset.pairBaseline, btn.dataset.pairFinetuned, btn.dataset.pairTitle || "");
    } else {
      open(btn.dataset.full, btn.dataset.caption || "");
    }
  });

  lbClose.addEventListener("click", close);
  lightbox.addEventListener("click", (e) => {
    if (e.target === lightbox) close();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !lightbox.hidden) close();
  });
})();
