// NeuroFocus 注意力热力图 —— 内容脚本
// 职责：
//   1) 在 B 站视频页右下角注入一个悬浮按钮，点击呼出侧边栏
//   2) 回答侧边栏的查询：当前 bvid / 标题 / 播放进度
//   3) 执行跳转：side panel 点击热力图格子 -> video.currentTime = t
//
// 注意：B 站是 SPA，路由切换不会重新加载页面，所以要定时确保按钮还在；
// 样式全部走 el.style（CSSOM），避免被页面 CSP 的 style-src 拦住。

(() => {
  if (window.__nfhInjected) return;
  window.__nfhInjected = true;

  const BTN_ID = "nfh-fab";
  let recording = false;
  let fabEl = null;
  let ringEl = null;

  // --------------------------------------------------------------------
  // 平台 / 播放器 / 标题
  // --------------------------------------------------------------------
  const isYouTube = () => /(^|\.)youtube\.com$/.test(location.hostname || "");

  // 只在「单个视频页」注入：B 站 /video/*，YouTube /watch?v=xxx
  // （YouTube Shorts 是 /shorts/xxx，这里刻意不注入）
  function isVideoPage() {
    const p = location.pathname || "";
    if (isYouTube()) {
      return p === "/watch" && new URLSearchParams(location.search).has("v");
    }
    return /^\/video\//.test(p);
  }

  function findVideo() {
    return (
      document.querySelector("#bilibili-player video") ||
      document.querySelector(".bpx-player-container video") ||
      document.querySelector(".bpx-player-video-wrap video") ||
      document.querySelector("#movie_player video.html5-main-video") ||
      document.querySelector("#movie_player video") ||
      document.querySelector("video.html5-main-video") ||
      document.querySelector("video")
    );
  }

  // 视频唯一 id：B 站用 BV 号，YouTube 加 "yt:" 前缀（存储 key 不会撞车）
  function videoKey() {
    if (isYouTube()) {
      const v = new URLSearchParams(location.search).get("v") || "";
      return v ? "yt:" + v : "";
    }
    const m = (location.pathname || "").match(/\/video\/(BV[0-9A-Za-z]+)/);
    return m ? m[1] : "";
  }

  function videoTitle() {
    const el =
      (isYouTube() &&
        (document.querySelector("#title h1 yt-formatted-string") ||
          document.querySelector("h1.ytd-watch-metadata yt-formatted-string") ||
          document.querySelector("ytd-watch-metadata #title"))) ||
      document.querySelector("#viewbox_report h1") ||
      document.querySelector("h1.video-title") ||
      document.querySelector(".video-title");
    if (el && el.textContent) return el.textContent.trim();
    return (document.title || "")
      .replace(/ - YouTube$/, "")
      .replace(/_哔哩哔哩.*$/, "")
      .trim();
  }

  function state() {
    const v = findVideo();
    const dur = v && isFinite(v.duration) ? v.duration : 0;
    return {
      ok: !!v,
      bvid: videoKey(),
      title: videoTitle(),
      currentTime: v ? v.currentTime : 0,
      duration: dur,
      paused: v ? v.paused : true,
      url: location.href,
    };
  }

  // --------------------------------------------------------------------
  // 悬浮按钮
  // --------------------------------------------------------------------
  function buildFab() {
    const host = document.createElement("div");
    host.id = BTN_ID;
    host.title = "注意力热力图 · 点击打开侧边栏";
    host.style.cssText = [
      "position:fixed",
      "right:22px",
      "bottom:110px",
      "z-index:2147483000",
      "width:52px",
      "height:52px",
      "border-radius:50%",
      "cursor:pointer",
      "display:flex",
      "align-items:center",
      "justify-content:center",
      "background:linear-gradient(150deg,#0f766e 0%,#15803d 100%)",
      "box-shadow:0 8px 22px rgba(0,0,0,.38),inset 0 0 0 1px rgba(255,255,255,.10)",
      "transition:transform .16s ease,box-shadow .16s ease",
      "user-select:none",
    ].join(";");

    // 3×3 方格 = 热力图意象。按 viewBox 36×36 严格居中排版：
    // 总宽 = 3*6.6 + 2*1.6 = 23，左右各留 (36-23)/2 = 6.5，格子中心正好落在 18。
    const CELL = 6.6;
    const GAP = 1.6;
    const PITCH = CELL + GAP;
    const PAD = (36 - (3 * CELL + 2 * GAP)) / 2;
    const cells = [];
    for (let i = 0; i < 9; i++) {
      const on = i % 3 === 1 || i === 7;
      const cx = PAD + (i % 3) * PITCH;
      const cy = PAD + Math.floor(i / 3) * PITCH;
      cells.push(
        `<rect x="${cx}" y="${cy}" width="${CELL}" height="${CELL}" rx="1.5" ` +
          `fill="${on ? "#c8ffd9" : "rgba(255,255,255,.34)"}"/>`
      );
    }

    const ring = document.createElement("span");
    ring.style.cssText = [
      "position:absolute",
      "inset:-6px",
      "border-radius:50%",
      "border:2px solid #34d399",
      "opacity:0",
      "pointer-events:none",
    ].join(";");

    host.innerHTML =
      '<svg width="30" height="30" viewBox="0 0 36 36" style="display:block" aria-hidden="true">' +
      cells.join("") +
      "</svg>";
    host.appendChild(ring);
    ringEl = ring;

    host.addEventListener("mouseenter", () => {
      host.style.transform = "translateY(-2px) scale(1.04)";
      host.style.boxShadow = "0 12px 26px rgba(0,0,0,.45),inset 0 0 0 1px rgba(255,255,255,.18)";
    });
    host.addEventListener("mouseleave", () => {
      host.style.transform = "";
      host.style.boxShadow = "0 8px 22px rgba(0,0,0,.38),inset 0 0 0 1px rgba(255,255,255,.10)";
    });
    host.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      chrome.runtime.sendMessage({ type: "nfh-open-panel" }, (res) => {
        if (chrome.runtime.lastError || !res || !res.ok) {
          host.title = "打开失败，请点浏览器右上角的扩展图标";
        }
      });
    });

    document.body.appendChild(host);
    fabEl = host;
    applyRecording();
    return host;
  }

  function applyRecording() {
    if (!fabEl) return;
    if (recording) {
      fabEl.style.background = "linear-gradient(150deg,#b91c1c 0%,#ea580c 100%)";
      if (ringEl) {
        ringEl.style.opacity = "0.85";
        ringEl.style.animation = "nfh-pulse 1.6s ease-out infinite";
      }
      fabEl.title = "注意力热力图 · 正在测量（点击打开侧边栏）";
    } else {
      fabEl.style.background = "linear-gradient(150deg,#0f766e 0%,#15803d 100%)";
      if (ringEl) {
        ringEl.style.opacity = "0";
        ringEl.style.animation = "none";
      }
      fabEl.title = "注意力热力图 · 点击打开侧边栏";
    }
  }

  function ensureKeyframes() {
    if (document.getElementById("nfh-kf")) return;
    const st = document.createElement("style");
    st.id = "nfh-kf";
    st.textContent =
      "@keyframes nfh-pulse{0%{transform:scale(.92);opacity:.9}70%{transform:scale(1.25);opacity:0}100%{opacity:0}}";
    (document.head || document.documentElement).appendChild(st);
  }

  function ensureFab() {
    if (!isVideoPage()) {
      if (fabEl && fabEl.isConnected) fabEl.remove();
      fabEl = null;
      return;
    }
    if (fabEl && fabEl.isConnected) return;
    if (!document.body) return;
    ensureKeyframes();
    buildFab();
  }

  // --------------------------------------------------------------------
  // 消息
  // --------------------------------------------------------------------
  chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
    if (!msg || !msg.type) return false;

    if (msg.type === "nfh-ping") {
      sendResponse({ ok: true, ready: true });
      return true;
    }

    if (msg.type === "nfh-state") {
      sendResponse(state());
      return true;
    }

    if (msg.type === "nfh-seek") {
      const v = findVideo();
      if (!v) {
        sendResponse({ ok: false, error: "没找到 video 元素" });
        return true;
      }
      // duration 在元数据加载前是 0/NaN，这时 (duration - 0.2) 会把时间夹成 0，
      // 所以有有效时长才夹上界。
      const want = Math.max(0, Number(msg.t) || 0);
      const dur = isFinite(v.duration) ? v.duration : 0;
      const t = dur > 0.5 ? Math.max(0, Math.min(want, dur - 0.2)) : want;
      try {
        v.currentTime = t;
      } catch (err) {
        sendResponse({ ok: false, error: String(err) });
        return true;
      }
      if (msg.play !== false) {
        const pr = v.play();
        if (pr && typeof pr.catch === "function") pr.catch(() => {});
      }
      sendResponse({ ok: true, currentTime: t });
      return true;
    }

    if (msg.type === "nfh-recording") {
      recording = !!msg.on;
      applyRecording();
      sendResponse({ ok: true });
      return true;
    }

    return false;
  });

  // --------------------------------------------------------------------
  // 启动 & SPA 路由
  // --------------------------------------------------------------------
  ensureKeyframes();
  ensureFab();
  setInterval(ensureFab, 1500);

  // 播放器被替换（换 P / 换视频）时，把引用清掉即可，findVideo 每次都重新查
  document.addEventListener(
    "click",
    () => {
      ensureFab();
    },
    true
  );
  // YouTube 是 SPA：站内跳转触发这个事件，不用等 1.5 s 的轮询
  window.addEventListener("yt-navigate-finish", () => {
    ensureFab();
  });
})();
