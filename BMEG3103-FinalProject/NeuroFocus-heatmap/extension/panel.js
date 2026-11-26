// NeuroFocus 注意力热力图 —— 侧边栏面板
//
// 数据流：
//   server.py（本地推理服务） --SSE--> 本页面
//   每条 score 到达时，向内容脚本询问 video.currentTime，
//   把 (当前播放秒 - 延迟补偿) 作为时间坐标，写进「秒 -> 分数」的原始表；
//   矩阵按当前「格宽」把秒聚合成格子再上色。
//
// 为什么按「秒」存原始数据：格宽可以随时改，不用重测。

const API = "http://127.0.0.1:8765";

const CFG = {
  high: ["#0e4429", "#006d32", "#26a641", "#39d353", "#7ef0a8"], // 集中（绿）
  low: ["#3a1d0a", "#5c2e0b", "#8a4a12", "#c26a1a", "#f0883e"], // 不集中（橙）
  empty: "#151b23",
  saveDebounce: 1200,
};

const S = {
  binSec: 10,
  lag: 2.0,
  thr: 0.5,
  thrDefault: 0.5,
  thrSource: "default",
  calibrated: false,
  models: [],       // GET /models 的原始列表
  modelId: "",      // 当前模型 id（也是存储 key 的一部分）
  calibLeft: 0,
  calibTotal: 30,
  calibTimer: null,
  bvid: "",
  title: "",
  duration: 0,
  cells: new Map(), // 秒 -> {sum, n, pass}
  pass: 1,
  es: null,
  connected: false,
  running: false,
  source: "",
  port: "",
  ports: [],
  ema: null,
  lastTime: 0,
  nowBin: -1,
  dirty: false,
  saveTimer: null,
  painted: new Map(),
  hintTimer: null,
  model: null,
  bridge: null,
  bridgeSeen: false,
  noisy: new Map(), // 秒 -> 1（这一秒 raw 有伪迹）
  marks: [],        // [{at: 视频秒数, t: Date.now()}]
};

const $ = (id) => document.getElementById(id);

/* ------------------------------------------------------------------ */
/* 小工具                                                              */
/* ------------------------------------------------------------------ */
function fmtTime(sec) {
  const s = Math.max(0, Math.round(sec || 0));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const x = s % 60;
  const mm = String(m).padStart(h ? 2 : 1, "0");
  const ss = String(x).padStart(2, "0");
  return h ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}

function hint(text, kind = "") {
  const el = $("hint");
  el.textContent = text || "";
  el.className = "hint " + kind;
  clearTimeout(S.hintTimer);
  if (text) S.hintTimer = setTimeout(() => { el.textContent = ""; }, 5000);
}

function setDot(cls, text) {
  $("dot").className = "dot " + cls;
  $("stateText").textContent = text;
}

/* ------------------------------------------------------------------ */
/* 与 B 站页面通信                                                      */
/* ------------------------------------------------------------------ */
const tabCache = { id: 0, t: 0 };

const SITE_URLS = ["*://www.bilibili.com/video/*", "*://www.youtube.com/watch*"];

async function getVideoState() {
  try {
    let id = tabCache.id;
    if (!id || Date.now() - tabCache.t > 5000) {
      const tabs = await chrome.tabs.query({ url: SITE_URLS });
      if (!tabs.length) return null;
      const tab = tabs.find((t) => t.active) || tabs[0];
      id = tab.id;
      tabCache.id = id;
      tabCache.t = Date.now();
    }
    const st = await chrome.tabs.sendMessage(id, { type: "nfh-state" });
    if (!st) return null;
    return { ...st, tabId: id };
  } catch (e) {
    tabCache.id = 0;
    return null;
  }
}

async function seekTo(t) {
  const st = await getVideoState();
  if (!st) {
    hint("没找到 B 站 / YouTube 视频页", "warn");
    return false;
  }
  try {
    await chrome.tabs.sendMessage(st.tabId, { type: "nfh-seek", t });
    return true;
  } catch (e) {
    hint("跳转失败：请刷新一下视频页（F5）", "warn");
    return false;
  }
}

function notifyRecording() {
  if (!tabCache.id) return;
  chrome.tabs
    .sendMessage(tabCache.id, { type: "nfh-recording", on: S.running })
    .catch(() => {});
}

/* ------------------------------------------------------------------ */
/* 本地服务（SSE）                                                      */
/* ------------------------------------------------------------------ */
function connectEvents() {
  if (S.es) {
    try { S.es.close(); } catch (e) {}
  }
  const es = new EventSource(API + "/events");
  S.es = es;

  es.addEventListener("open", () => setConnected(true));
  es.addEventListener("error", () => setConnected(false));
  es.addEventListener("message", (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch (e) { return; }
    onEvent(msg);
  });
}

function setConnected(on) {
  if (S.connected === on) return;
  S.connected = on;
  if (!on) {
    S.running = false;
    setDot("off", "本地服务未连接");
    $("subLine").textContent = `本地推理服务没在运行：双击 heatmap\\run.bat 启动，或前台 py -3 server.py（${API}）`;
    stopCalibCountdown();
    renderBadge();
    syncButtons();
  } else {
    setDot("on", "已连接");
  }
}

function onEvent(msg) {
  switch (msg.type) {
    case "state": {
      setConnected(true);
      S.running = !!msg.running;
      S.source = msg.source || "";
      if (typeof msg.thr === "number") S.thr = msg.thr;
      // 模型身份变了：先把旧模型的格子落盘，再换 key 重新载入本视频在新模型下的数据
      if (msg.model) adoptModel(msg.model);
      setDot(S.running ? "live" : "on", S.running ? "测量中" : "已连接 · 待机");
      // adoptModel 里有 await，这一行不能读 S.model（可能还停在旧模型上）
      const m = msg.model || S.model;
      const parts = [];
      if (m) parts.push(`模型 ${m.id} · ${m.params} 参数 · ${m.features} 维`);
      parts.push(`阈值 ${S.thr.toFixed(2)}`);
      if (S.running && S.source) {
        const label = { sim: "模拟数据（演示）", serial: "头环串口", replay: "回放" }[S.source] || S.source;
        parts.push(msg.port ? `${label} ${msg.port}` : label);
      }
      $("subLine").textContent = parts.join(" · ");
      syncButtons();
      notifyRecording();
      break;
    }
    case "score":
      onScore(msg);
      break;
    case "calib":
      if (msg.done) {
        if (typeof msg.thr === "number") S.thr = msg.thr;
        if (typeof msg.calibrated === "boolean") S.calibrated = msg.calibrated;
        stopCalibCountdown();
        renderBadge();
        renderCalib();
        renderGauge();
        hint(msg.text || "校准完成", msg.ok === false ? "warn" : "ok");
      } else {
        startCalibCountdown(msg.seconds || 30);
        hint(msg.text || `静息校准中…`, "");
      }
      break;
    case "log":
      hint(msg.text || "", msg.level === "error" ? "warn" : "");
      break;
    case "bridge":
      onBridge(msg);
      break;
    default:
      break;
  }
}

function syncButtons() {
  $("btnStart").disabled = !S.connected || S.running;
  $("btnStop").disabled = !S.connected || !S.running;
  $("btnCalib").disabled = !S.connected || !S.running;
  $("btnStart").textContent = S.running ? "测量中…" : "开始测量";
  // 换模型要停止之后才能点：服务端也会拒绝，这里先拦一道，省得点了没反应
  $("modelSel").disabled = !S.connected || S.running;
  $("btnModels").disabled = !S.connected;
  $("modelSel").title = S.running
    ? "测量中不能换模型：先停止再切换"
    : "当前模型。切换后立刻按该模型的阈值生效。";
}

async function post(path, body) {
  const res = await fetch(API + path, {
    method: "POST",
    headers: { "Content-Type": "text/plain" },
    body: JSON.stringify(body || {}),
  });
  return res.json();
}

/* ------------------------------------------------------------------ */
/* 模型与校准                                                          */
/* ------------------------------------------------------------------ */
// 服务端报来的模型就是真相：面板不自己猜 id，只负责把界面刷新到跟它一致。
async function adoptModel(m) {
  if (!m || !m.id) return;
  const changed = !!S.modelId && m.id !== S.modelId;
  if (changed) await flushSave();   // 旧模型的格子先落盘（此刻 key 还是旧的）
  S.model = m;
  S.modelId = m.id;
  if (typeof m.thr === "number") S.thr = m.thr;
  if (typeof m.thr_default === "number") S.thrDefault = m.thr_default;
  S.thrSource = m.thr_source || "default";
  S.calibrated = !!m.calibrated;
  if (changed) {
    if (!S.running) await reloadVideoData();
    savePrefs();
  }
  syncModelSel();
  renderModelBox();
  renderBadge();
  renderCalib();
  renderGauge();
}

// 换模型后，本视频的数据要换成「新模型」那一份：同一个视频，两个模型各记各的
async function reloadVideoData() {
  S.cells.clear();
  S.noisy.clear();
  S.marks = [];
  S.pass = 1;
  S.painted.clear();
  S.nowBin = -1;
  S.lastTime = 0;
  const saved = S.bvid ? await loadVideo(S.bvid) : null;
  applySaved(saved);
  renderMarks();
  rebuildGrid();
}

function applySaved(saved) {
  if (saved && saved.cells) {
    for (const [sec, arr] of Object.entries(saved.cells)) {
      S.cells.set(Number(sec), { sum: arr[0], n: arr[1], pass: arr[2] || 1 });
    }
    S.pass = 1 + Math.max(0, ...Array.from(S.cells.values(), (r) => (r.pass || 1) - 1));
  }
  if (saved && Array.isArray(saved.noisy)) {
    for (const sec of saved.noisy) S.noisy.set(Number(sec), 1);
  }
  if (saved && Array.isArray(saved.marks)) {
    S.marks = saved.marks
      .map((m) => ({ at: Number(m[0]) || 0, t: Number(m[1]) || 0 }))
      .sort((a, b) => a.at - b.at);
  }
}

function renderModelBox() {
  const m = S.model;
  $("modelNow").textContent = m ? m.id : "未连接";
  if (!m) {
    $("modelInfo").textContent = "—";
    return;
  }
  const bits = [`${m.params} 参数`, `${m.features} 维`, `${m.window}s 窗`];
  if (m.acc != null) bits.push(`acc ${m.acc.toFixed(3)}`);
  if (m.trained_at) bits.push(m.trained_at);
  bits.push(`阈值 ${Number(m.thr).toFixed(2)}`);
  $("modelInfo").textContent = bits.join(" · ");
}

// 仪表盘角标：区分「这个模型校准过」和「只是在用它的默认阈值」
function renderBadge() {
  const b = $("thrBadge");
  if (!S.connected) {
    b.className = "badge";
    b.textContent = "未连接";
    return;
  }
  if (S.calibrated) {
    b.className = "badge calib";
    b.textContent = `已校准 ${S.thr.toFixed(2)}`;
  } else {
    b.className = "badge";
    b.textContent = `未校准 · 默认 ${S.thrDefault.toFixed(2)}`;
  }
}

function renderCalib() {
  const box = $("calibInfo");
  const track = $("calibTrack");
  const on = S.calibLeft > 0;
  track.hidden = !on;
  if (on) {
    const frac = 1 - S.calibLeft / Math.max(1, S.calibTotal);
    $("calibFill").style.width = `${Math.max(0, Math.min(1, frac)) * 100}%`;
    box.textContent = `校准中：还剩 ${S.calibLeft}s（保持静息，别动，别说话）`;
    return;
  }
  if (!S.connected) {
    box.textContent = "—";
  } else if (S.calibrated) {
    box.textContent = `已校准：阈值 ${S.thr.toFixed(2)}（当前模型，已保存）`;
  } else {
    box.textContent =
      `未校准：正在用这个模型自带的默认阈值 ${S.thrDefault.toFixed(2)}，建议先静息校准 30 秒`;
  }
}

function startCalibCountdown(sec) {
  stopCalibCountdown();
  S.calibTotal = Math.max(1, Number(sec) || 30);
  S.calibLeft = S.calibTotal;
  renderCalib();
  S.calibTimer = setInterval(() => {
    S.calibLeft = Math.max(0, S.calibLeft - 1);
    renderCalib();
    if (!S.calibLeft) stopCalibCountdown();
  }, 1000);
}

function stopCalibCountdown() {
  clearInterval(S.calibTimer);
  S.calibTimer = null;
  S.calibLeft = 0;
  renderCalib();
}

async function refreshModels() {
  try {
    const r = await fetch(API + "/models", { cache: "no-store" }).then((x) => x.json());
    S.models = Array.isArray(r.models) ? r.models : [];
    renderModelOptions();
    const ok = S.models.filter((m) => m.ok).length;
    hint(`模型目录 ${S.models.length} 个文件，可用 ${ok} 个`, ok ? "ok" : "warn");
    return r;
  } catch (e) {
    hint("连不上本地服务，模型列表没刷新", "warn");
    return null;
  }
}

function renderModelOptions() {
  const sel = $("modelSel");
  sel.textContent = "";
  if (!S.models.length) {
    const o = document.createElement("option");
    o.value = "";
    o.textContent = "（模型目录是空的）";
    o.disabled = true;
    sel.appendChild(o);
    return;
  }
  for (const m of S.models) {
    const o = document.createElement("option");
    o.value = m.id;
    const when = m.trained_at ? " · " + m.trained_at : "";
    o.textContent = m.ok ? `${m.id}${when}` : `${m.id}（不可用）`;
    o.title = m.ok
      ? `${m.params} 参数 · ${m.features} 维${m.acc != null ? " · acc " + m.acc.toFixed(3) : ""}`
      : m.reason;
    o.disabled = !m.ok;     // 不兼容的模型留在列表里但点不了，鼠标悬停能看到原因
    sel.appendChild(o);
  }
  // 当前模型可能是用 --model 直接指的文件，不在目录里
  if (S.modelId && !S.models.some((m) => m.id === S.modelId)) {
    const o = document.createElement("option");
    o.value = S.modelId;
    o.textContent = `${S.modelId}（不在模型目录里）`;
    sel.appendChild(o);
  }
  syncModelSel();
}

function syncModelSel() {
  const sel = $("modelSel");
  if (!sel) return;
  if (S.modelId && Array.from(sel.options).some((o) => o.value === S.modelId)) {
    sel.value = S.modelId;
  }
}

async function switchModel() {
  const id = $("modelSel").value;
  if (!id || id === S.modelId) return;
  if (S.running) {
    hint("测量中不能换模型：先停止，再切换", "warn");
    syncModelSel();
    return;
  }
  await flushSave();
  const r = await post("/model", { id }).catch(() => null);
  if (!r || !r.ok) {
    hint((r && r.error) || "切换失败", "warn");
    syncModelSel();
    return;
  }
  await adoptModel(r.model);
  hint(`已切到 ${r.model.id}，阈值 ${Number(r.model.thr).toFixed(2)}`, "ok");
}


/* ------------------------------------------------------------------ */
/* 分数 -> 时间格                                                       */
/* ------------------------------------------------------------------ */
async function onScore(m) {
  S.ema = typeof m.ema === "number" ? m.ema : m.p;
  if (typeof m.thr === "number") S.thr = m.thr;
  renderGauge();

  const st = await getVideoState();
  if (!st || !st.ok) return;
  syncVideo(st);
  if (st.paused) return;
  if (!m.ok) {
    hint("信号质量差，这一秒不计入（信号码 " + m.signal + "）", "warn");
    return;
  }

  const t = st.currentTime - S.lag;
  if (t < 0) return;
  addSample(Math.floor(t), m.p);
}

function addSample(sec, p) {
  let rec = S.cells.get(sec);
  // 新一轮（点了格子 / 往回拖动）重新测到的秒，直接覆盖旧数据，
  // 否则同一格里会混进上一轮的分数。
  if (!rec || (rec.pass || 1) !== S.pass) rec = { sum: 0, n: 0, pass: S.pass };
  rec.sum += p;
  rec.n += 1;
  rec.pass = S.pass;
  S.cells.set(sec, rec);
  scheduleSave();
  paintBin(Math.floor(sec / S.binSec));
}

/* ------------------------------------------------------------------ */
/* Arduino 桥（方案 C）：状态 / 按键标记 / 伪迹 / LED 反馈              */
/* ------------------------------------------------------------------ */
async function onBridge(m) {
  S.bridge = m;
  S.bridgeSeen = true;

  if (m.mark) {
    // 物理按键按下：毫秒级事件，立刻记到当前视频时间上
    await addMarkNow("按键标记");
  }

  if (m.artifact) {
    const st = await getVideoState();
    if (st && st.ok && !st.paused) {
      const sec = Math.floor(st.currentTime - S.lag);
      if (sec >= 0) addNoisy(sec);
    }
  }

  renderBridge();
}

function renderBridge() {
  const b = S.bridge;
  const bar = $("bridgeBar");
  if (!b) {
    bar.hidden = true;
    return;
  }
  bar.hidden = false;
  const link = !!b.link;
  $("brDot").className = "dot " + (link ? "on" : "off");
  $("brText").textContent = "Arduino 桥" + (b.synthetic ? "（模拟）" : "");
  const bits = [link ? "TGAM link ✓" : "TGAM 未连接"];
  if (b.signal != null) bits.push("signal " + b.signal);
  if (b.raw_std) bits.push("raw_std " + b.raw_std);
  if (b.bad) bits.push("坏帧 " + b.bad);
  $("brMeta").textContent = bits.join(" · ");
  $("btnLed").textContent = b.led ? "LED 开" : "LED 关";
  const canCmd = S.connected && S.running && S.source === "serial";
  $("btnLed").disabled = !canCmd;
  $("btnPing").disabled = !canCmd;
  $("btnLed").title = canCmd
    ? "发 #LED:n，让 Arduino 上的 LED 做本地反馈"
    : "只在串口数据源、测量中可用";
}

function addNoisy(sec) {
  if (S.noisy.has(sec)) return;
  S.noisy.set(sec, 1);
  scheduleSave();
  paintBin(Math.floor(sec / S.binSec));
}

async function renderMarks() {
  const box = $("marks");
  box.textContent = "";
  box.hidden = S.marks.length === 0;
  S.marks.forEach((m) => {
    const b = document.createElement("button");
    b.className = "chip";
    b.textContent = "⚑ " + fmtTime(m.at);
    b.title = "跳到 " + fmtTime(m.at) + " 重测这一段";
    b.addEventListener("click", async () => {
      newPass("按标记重测");
      clearBin(Math.floor(m.at / S.binSec));
      S.lastTime = m.at;
      await seekTo(m.at);
    });
    box.appendChild(b);
  });
}

// 标记：以前只有 Arduino 那个物理按键能打，现在面板按钮 / 快捷键 M 也能打
async function addMarkNow(src = "标记") {
  const st = await getVideoState();
  if (!st || !st.ok) {
    hint("没找到视频页，标记不成", "warn");
    return;
  }
  const at = Math.max(0, st.currentTime - S.lag);
  S.marks.push({ at, t: Date.now() });
  S.marks.sort((a, b) => a.at - b.at);
  renderMarks();
  scheduleSave();
  hint(`${src} ${fmtTime(at)}（点标签可回看）`, "ok");
}

async function toggleLed() {
  const want = !(S.bridge && S.bridge.led);
  const r = await post("/bridge", { cmd: want ? "#LED:1" : "#LED:0" });
  if (!r || !r.ok) hint((r && r.error) || "命令没发出去", "warn");
  else hint(want ? "Arduino LED 反馈已开" : "Arduino LED 反馈已关", "ok");
}

async function pingBridge() {
  const r = await post("/bridge", { cmd: "#PING" });
  if (!r || !r.ok) hint((r && r.error) || "命令没发出去", "warn");
  else hint("已向 Arduino 发 #PING", "ok");
}

// 开始重测某一格前，先把这格里的旧秒清掉，保证一格内的分数都来自同一轮
function clearBin(i) {
  const from = i * S.binSec;
  const to = from + S.binSec;
  let hit = false;
  for (const sec of Array.from(S.cells.keys())) {
    if (sec >= from && sec < to) {
      S.cells.delete(sec);
      hit = true;
    }
  }
  if (hit) {
    scheduleSave();
    paintBin(i);
  }
  return hit;
}

function newPass(reason) {
  S.pass += 1;
  if (reason) hint(reason + `（第 ${S.pass} 轮）`);
}

/* ------------------------------------------------------------------ */
/* 视频切换 / 轮询                                                      */
/* ------------------------------------------------------------------ */
async function syncVideo(st) {
  if (!st.bvid) return;
  if (st.bvid !== S.bvid) {
    await flushSave();
    S.bvid = st.bvid;
    S.title = st.title || "";
    S.duration = st.duration || 0;
    S.cells.clear();
    S.pass = 1;
    S.nowBin = -1;
    S.lastTime = 0;
    S.painted.clear();
    const saved = await loadVideo(S.bvid);
    S.noisy.clear();
    S.marks = [];
    applySaved(saved);
    renderMarks();
    renderBridge();
    $("videoLine").textContent = `${vidLabel(S.bvid)} · ${S.title}`.slice(0, 96);
    rebuildGrid();
  } else {
    if (st.duration && Math.abs(st.duration - S.duration) > 0.5) {
      S.duration = st.duration;
      rebuildGrid();
    }
    if (st.title && st.title !== S.title) {
      S.title = st.title;
      $("videoLine").textContent = `${vidLabel(S.bvid)} · ${S.title}`.slice(0, 96);
    }
  }
}

async function pollVideo() {
  const st = await getVideoState();
  if (!st || !st.ok) {
    $("videoLine").textContent = "未找到视频播放器 —— 请打开 B 站 / YouTube 视频页";
    return;
  }
  await syncVideo(st);

  const bin = S.binSec ? Math.floor((st.currentTime - S.lag) / S.binSec) : -1;
  if (bin !== S.nowBin) {
    paintNow(S.nowBin, bin);
    S.nowBin = bin;
  }

  if (S.lastTime && !st.paused && st.currentTime + 3 < S.lastTime) {
    newPass("检测到往回拖动");
  }
  if (!st.paused) S.lastTime = st.currentTime;
}

/* ------------------------------------------------------------------ */
/* 绘制                                                                */
/* ------------------------------------------------------------------ */
function binCount() {
  if (!S.duration || !S.binSec) return 0;
  return Math.max(1, Math.min(4000, Math.ceil(S.duration / S.binSec)));
}

function aggregate() {
  const n = binCount();
  const out = new Array(n);
  for (let i = 0; i < n; i++) out[i] = { sum: 0, n: 0, pass: 0, noisy: 0 };
  if (!n) return out;
  for (const [sec, rec] of S.cells) {
    const b = Math.floor(sec / S.binSec);
    if (b < 0 || b >= n) continue;
    out[b].sum += rec.sum;
    out[b].n += rec.n;
    if ((rec.pass || 1) > out[b].pass) out[b].pass = rec.pass || 1;
  }
  for (const sec of S.noisy.keys()) {
    const b = Math.floor(sec / S.binSec);
    if (b >= 0 && b < n) out[b].noisy += 1;
  }
  return out;
}

function colorFor(p) {
  if (p == null) return CFG.empty;
  const thr = Math.max(0.05, Math.min(0.95, S.thr));
  if (p >= thr) {
    const k = Math.min(4, Math.floor(((p - thr) / Math.max(1e-6, 1 - thr)) * 5));
    return CFG.high[k];
  }
  const k = Math.min(4, Math.floor((p / Math.max(1e-6, thr)) * 5));
  return CFG.low[k];
}

function rebuildGrid() {
  const n = binCount();
  const m = $("matrix");
  if (m.childElementCount !== n) {
    m.textContent = "";
    const frag = document.createDocumentFragment();
    for (let i = 0; i < n; i += 1) {
      const d = document.createElement("div");
      d.className = "cell";
      d.dataset.i = String(i);
      frag.appendChild(d);
    }
    m.appendChild(frag);
    S.painted.clear();
    $("matrixWrap").scrollTop = 0;
  }
  paintAll();
  $("rangeLine").textContent = S.duration
    ? `${fmtTime(0)} – ${fmtTime(S.duration)} · ${n} 格 / ${S.binSec}s`
    : "—";
}

function paintAll() {
  const agg = aggregate();
  const m = $("matrix");
  for (let i = 0; i < m.childElementCount && i < agg.length; i += 1) {
    const a = agg[i];
    const p = a.n ? a.sum / a.n : null;
    const col = colorFor(p);
    const cell = m.children[i];
    if (S.painted.get(i) !== col) {
      cell.style.background = col;
      S.painted.set(i, col);
    }
    const multi = a.pass > 1;
    if (cell.classList.contains("multi") !== multi) cell.classList.toggle("multi", multi);
    const nois = (a.noisy || 0) > 0;
    if (cell.classList.contains("noisy") !== nois) cell.classList.toggle("noisy", nois);
    cell.dataset.p = p == null ? "" : (p * 100).toFixed(0);
    cell.dataset.n = String(a.n);
    cell.dataset.pass = String(a.pass);
    cell.dataset.noisy = String(a.noisy || 0);
  }
}

function paintBin(i) {
  if (i < 0) return;
  const m = $("matrix");
  if (i >= m.childElementCount) return;
  const agg = aggregate();
  const a = agg[i];
  if (!a) return;
  const p = a.n ? a.sum / a.n : null;
  const col = colorFor(p);
  const cell = m.children[i];
  if (S.painted.get(i) !== col) {
    cell.style.background = col;
    S.painted.set(i, col);
  }
  cell.dataset.p = p == null ? "" : (p * 100).toFixed(0);
  cell.dataset.n = String(a.n);
  cell.dataset.pass = String(a.pass);
  if (cell.classList.contains("multi") !== a.pass > 1) cell.classList.toggle("multi", a.pass > 1);
  const nois = (a.noisy || 0) > 0;
  if (cell.classList.contains("noisy") !== nois) cell.classList.toggle("noisy", nois);
  cell.dataset.noisy = String(a.noisy || 0);
}

function paintNow(prev, next) {
  const m = $("matrix");
  if (prev >= 0 && prev < m.childElementCount) m.children[prev].classList.remove("now");
  if (next >= 0 && next < m.childElementCount) {
    const cell = m.children[next];
    cell.classList.add("now");
    if (S.painted.size > 0) cell.scrollIntoView({ block: "nearest" });
  }
}

function renderGauge() {
  const p = S.ema;
  $("pct").textContent = p == null ? "--" : String(Math.round(p * 100));
  const fill = $("barFill");
  fill.style.width = p == null ? "0%" : `${Math.max(0, Math.min(1, p)) * 100}%`;
  const focus = p != null && p >= S.thr;
  fill.style.background = focus
    ? "linear-gradient(90deg,#15803d,#4ade80)"
    : "linear-gradient(90deg,#8a4a12,#f0883e)";
  $("thrMark").style.left = `${S.thr * 100}%`;
  const v = $("verdict");
  v.textContent = p == null ? "—" : focus ? "专注 FOCUS" : "静息 REST";
  v.className = "verdict " + (p == null ? "" : focus ? "focus" : "rest");
}

/* ------------------------------------------------------------------ */
/* 存储                                                                */
/* ------------------------------------------------------------------ */
// 存储 key 带上模型 id：同一个视频、两个模型的数据各存各的，换模型不会互相覆盖。
// 旧的 nfh:v1:<bvid> 保持原样不动——新版不读也不删，万一要翻旧数据还在。
const keyFor = (bvid) => `nfh:v2:${S.modelId || "default"}:${bvid}`;

// 视频 id 展示：B 站存 BV 号，YouTube 存 "yt:xxxx"，这里转成人看得懂的标签
function vidLabel(id) {
  if (!id) return "";
  if (id.startsWith("yt:")) return "YouTube · " + id.slice(3);
  return "B站 · " + id;
}

async function loadVideo(bvid) {
  try {
    const o = await chrome.storage.local.get(keyFor(bvid));
    return o[keyFor(bvid)] || null;
  } catch (e) {
    return null;
  }
}

function scheduleSave() {
  S.dirty = true;
  clearTimeout(S.saveTimer);
  S.saveTimer = setTimeout(flushSave, CFG.saveDebounce);
}

async function flushSave() {
  if (!S.dirty || !S.bvid) return;
  S.dirty = false;
  const cells = {};
  for (const [sec, r] of S.cells) {
    cells[sec] = [Math.round(r.sum * 1000) / 1000, r.n, r.pass || 1];
  }
  const noisy = Array.from(S.noisy.keys()).sort((a, b) => a - b);
  const marks = S.marks.map((m) => [Math.round(m.at * 10) / 10, m.t]);
  try {
    await chrome.storage.local.set({
      [keyFor(S.bvid)]: { binSec: S.binSec, cells, noisy, marks, savedAt: Date.now() },
    });
  } catch (e) {
    hint("保存失败（存储空间可能已满）", "warn");
  }
}

async function savePrefs() {
  try {
    await chrome.storage.local.set({
      "nfh:prefs": { binSec: S.binSec, lag: S.lag, model: S.modelId },
    });
  } catch (e) {}
}

/* ------------------------------------------------------------------ */
/* 交互                                                                */
/* ------------------------------------------------------------------ */
/* ---------------- 串口（硬件接入） ---------------- */
function syncPortRow() {
  $("portRow").hidden = $("srcSel").value !== "serial";
}

async function refreshPorts() {
  const inp = $("portSel");
  let ports = [];
  let def = "COM7";
  try {
    const r = await fetch(API + "/ports", { cache: "no-store" }).then((x) => x.json());
    if (r && r.ok) {
      if (Array.isArray(r.ports)) ports = r.ports;
      if (r.default) def = r.default;
    }
  } catch (e) {
    hint("连不上本地服务，无法扫描串口", "warn");
  }
  S.ports = ports;
  const list = $("portList");
  list.innerHTML = "";
  for (const p of ports) {
    const o = document.createElement("option");
    o.value = p.device;
    if (p.description && p.description !== "n/a") o.label = p.description;
    list.appendChild(o);
  }
  // 已有值时不动它（用户可能手输了 COM 口）；否则退到扫描结果 / 记忆值 / 服务端默认
  if (!inp.value) inp.value = S.port || (ports[0] && ports[0].device) || def;
  S.port = inp.value;
}

function bind() {
  $("btnStart").addEventListener("click", async () => {
    const source = $("srcSel").value;
    const body = { source };
    if (source === "serial") {
      const port = $("portSel").value.trim();
      if (port) body.serial_port = port; // 留空则由服务端回退到默认 COM 口
    }
    try {
      const r = await post("/start", body);
      if (!r.ok) hint(r.error || "启动失败", "warn");
      else hint(`已请求启动（${source}${body.serial_port ? " " + body.serial_port : ""}）`, "ok");
    } catch (e) {
      hint("连接不到本地服务，先跑 server.py", "warn");
    }
  });

  $("btnPorts").addEventListener("click", async () => {
    await refreshPorts();
    hint(S.ports.length ? `发现 ${S.ports.length} 个串口` : "没有发现串口", S.ports.length ? "ok" : "warn");
  });

  $("btnLed").addEventListener("click", toggleLed);
  $("btnPing").addEventListener("click", pingBridge);

  $("portSel").addEventListener("input", () => {
    S.port = $("portSel").value.trim();
    chrome.storage.local.set({ "nfh:port": S.port }).catch(() => {});
  });

  $("btnStop").addEventListener("click", async () => {
    try { await post("/stop", {}); } catch (e) {}
    await flushSave();
  });

  $("btnCalib").addEventListener("click", async () => {
    try {
      const r = await post("/calibrate", {});
      if (!r.ok) hint(r.error || "校准失败", "warn");
      else {
        startCalibCountdown(r.seconds || 30);
        hint(`静息校准开始：保持闭眼放松 ${r.seconds || 30} 秒`, "");
      }
    } catch (e) {
      hint("连接不到本地服务", "warn");
    }
  });

  $("modelSel").addEventListener("change", switchModel);
  $("btnModels").addEventListener("click", async () => {
    await refreshModels();
  });
  $("btnMark").addEventListener("click", () => addMarkNow("标记"));

  // 看视频时手一般不在侧边栏上，给个快捷键（M = mark）
  document.addEventListener("keydown", (e) => {
    if (e.key !== "m" && e.key !== "M") return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const t = e.target;
    if (t && /^(INPUT|SELECT|TEXTAREA)$/.test(t.tagName)) return;
    e.preventDefault();
    addMarkNow("标记");
  });

  $("binSel").addEventListener("change", (e) => {
    S.binSec = Number(e.target.value) || 10;
    rebuildGrid();
    savePrefs();
  });

  $("lagRange").addEventListener("input", (e) => {
    S.lag = Number(e.target.value);
    $("lagVal").textContent = S.lag.toFixed(1) + "s";
    paintNow(S.nowBin, -1);
    S.nowBin = -1;
    savePrefs();
  });

  $("srcSel").addEventListener("change", (e) => {
    chrome.storage.local.set({ "nfh:source": e.target.value }).catch(() => {});
    syncPortRow();
    if (e.target.value === "serial") refreshPorts();
  });

  $("btnExport").addEventListener("click", exportCsv);

  $("btnClear").addEventListener("click", async () => {
    if (!S.bvid) return;
    if (!confirm(`清空 ${vidLabel(S.bvid)} 的全部注意力记录？`)) return;
    S.cells.clear();
    S.noisy.clear();
    S.marks = [];
    S.pass = 1;
    S.painted.clear();
    renderMarks();
    await chrome.storage.local.remove(keyFor(S.bvid));
    rebuildGrid();
    hint("已清空", "ok");
  });

  const matrix = $("matrix");

  matrix.addEventListener("click", async (e) => {
    const cell = e.target.closest(".cell");
    if (!cell) return;
    const i = Number(cell.dataset.i);
    const t = i * S.binSec + 0.2;
    newPass("点击格子 → 重新测量");
    const ok = await seekTo(t);
    if (!ok) {
      hint("跳转失败：这一页可能没找到播放器", "warn");
      return;
    }
    clearBin(i); // 跳转成功才清空旧数据，失败时保留原本的热力图
    S.lastTime = t; // 自己跳的，不算“往回拖动”，否则会再开一轮
    hint(`已跳到 ${fmtTime(t)}，新一轮测量从这一刻开始`, "ok");
    paintNow(S.nowBin, i);
    S.nowBin = i;
  });

  matrix.addEventListener("mousemove", (e) => {
    const cell = e.target.closest(".cell");
    const tip = $("tip");
    if (!cell) {
      tip.classList.remove("on");
      return;
    }
    const i = Number(cell.dataset.i);
    const n = Number(cell.dataset.n || 0);
    const p = cell.dataset.p;
    const pass = Number(cell.dataset.pass || 0);
    const from = i * S.binSec;
    const to = from + S.binSec;
    const passTxt = pass > 1 ? ` · 第 ${pass} 轮（重测过）` : "";
    const nois = Number(cell.dataset.noisy || 0);
    const noisTxt = nois ? ` · 含 ${nois}s 伪迹（接触/眼动，参考着看）` : "";
    tip.innerHTML =
      `<b>${fmtTime(from)} – ${fmtTime(to)}</b><br>` +
      (n
        ? `平均 P(专注) ≈ ${p}% · ${n} 个窗${passTxt}${noisTxt}`
        : "尚未测量 · 点击跳到这里开始测");
    tip.classList.add("on");
    const x = Math.min(window.innerWidth - 232, e.clientX + 12);
    const y = Math.min(window.innerHeight - 70, e.clientY + 14);
    tip.style.left = Math.max(4, x) + "px";
    tip.style.top = Math.max(4, y) + "px";
  });

  matrix.addEventListener("mouseleave", () => $("tip").classList.remove("on"));

  window.addEventListener("beforeunload", flushSave);
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden") flushSave();
  });
}

function exportCsv() {
  const agg = aggregate();
  const rows = ["video_id,bin_start_s,bin_end_s,mean_p,samples,pass,noisy"];
  agg.forEach((a, i) => {
    if (!a.n) return;
    rows.push(
      [
        S.bvid,
        i * S.binSec,
        (i + 1) * S.binSec,
        (a.sum / a.n).toFixed(4),
        a.n,
        a.pass || 1,
        a.noisy || 0,
      ].join(",")
    );
  });
  if (rows.length === 1) {
    hint("还没有数据可导出", "warn");
    return;
  }
  const blob = new Blob(["\ufeff" + rows.join("\r\n")], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `nfh_${(S.modelId || "model").replace(/[^0-9A-Za-z_-]/g, "_")}_${(
    S.bvid || "video"
  ).replace(/[^0-9A-Za-z_-]/g, "_")}_${S.binSec}s.csv`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
  hint(`已导出 ${rows.length - 1} 行`, "ok");
}

/* ------------------------------------------------------------------ */
/* 启动                                                                */
/* ------------------------------------------------------------------ */
async function init() {
  try {
    const pref = await chrome.storage.local.get(["nfh:prefs", "nfh:source", "nfh:port"]);
    const p = pref["nfh:prefs"] || {};
    if (p.binSec) S.binSec = Number(p.binSec);
    if (typeof p.lag === "number") S.lag = Number(p.lag);
    if (p.model) S.modelId = String(p.model);
    if (pref["nfh:source"]) $("srcSel").value = pref["nfh:source"];
    if (pref["nfh:port"]) S.port = String(pref["nfh:port"]);
  } catch (e) {}

  $("binSel").value = String(S.binSec);
  $("lagRange").value = String(S.lag);
  $("lagVal").textContent = S.lag.toFixed(1) + "s";

  bind();
  syncPortRow();
  if ($("srcSel").value === "serial") refreshPorts();
  renderGauge();
  renderBridge();
  renderMarks();
  renderModelBox();
  renderBadge();
  renderCalib();
  setDot("off", "本地服务未连接");
  connectEvents();
  refreshModels();
  pollVideo();
  setInterval(pollVideo, 1000);
}

init();
