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

    const cards = [
      { label: `${stem} • RGB Input`, file: `assets/images/qualitative/input_${stem}.png` },
      { label: `${stem} • Baseline Det`, file: `assets/images/qualitative/detection_baseline_${stem}.png` },
      { label: `${stem} • Fine-tuned Det`, file: `assets/images/qualitative/detection_finetuned_${stem}.png` },
      { label: `${stem} • Baseline Seg`, file: `assets/images/qualitative/seg_baseline_${stem}.png` },
      { label: `${stem} • Fine-tuned Seg`, file: `assets/images/qualitative/seg_finetuned_${stem}.png` },
    ];

    cards.forEach((card) => {
      const wrapper = document.createElement("div");
      wrapper.className = "qual-card";
      wrapper.innerHTML = `
        <p>${card.label}</p>
        <img src="${card.file}" alt="${card.label}" loading="lazy" />
      `;
      row.appendChild(wrapper);
    });

    grid.appendChild(row);
  });
}

async function loadResults() {
  try {
    const response = await fetch("assets/data/results.json");
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
      map50: Number(models[key].mAP50 || 0),
      map95: Number(models[key].mAP50_95 || 0),
      fps: Number(models[key].FPS || 0),
      paramsM: Number(models[key].params || 0) / 1e6,
      flopsG: Number(models[key].FLOPs || 0) / 1e9,
    }));

  if (!rows.length) {
    body.innerHTML = '<tr><td colspan="6" class="notice">benchmark_zeroshot.json not found.</td></tr>';
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
    const response = await fetch("assets/data/benchmark_zeroshot.json");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    renderZeroShotTable(data);
  } catch (err) {
    console.error(err);
    document.getElementById("zeroshot-body").innerHTML =
      '<tr><td colspan="6" class="notice">benchmark_zeroshot.json not found.</td></tr>';
  }
}

loadResults();
loadZeroShot();

// Teaser lightbox
(function () {
  const lightbox = document.getElementById("lightbox");
  const lbImg = document.getElementById("lightbox-img");
  const lbCaption = document.getElementById("lightbox-caption");
  const lbClose = document.getElementById("lightbox-close");
  if (!lightbox || !lbImg || !lbCaption || !lbClose) return;

  function open(src, caption) {
    lbImg.src = src;
    lbImg.alt = caption;
    lbCaption.textContent = caption;
    lightbox.hidden = false;
    document.body.style.overflow = "hidden";
  }

  function close() {
    lightbox.hidden = true;
    document.body.style.overflow = "";
  }

  document.querySelectorAll(".teaser-zoom").forEach((btn) => {
    btn.addEventListener("click", () => open(btn.dataset.full, btn.dataset.caption || ""));
  });

  lbClose.addEventListener("click", close);
  lightbox.addEventListener("click", (e) => {
    if (e.target === lightbox) close();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !lightbox.hidden) close();
  });
})();
