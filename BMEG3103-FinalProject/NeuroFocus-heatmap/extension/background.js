// NeuroFocus 注意力热力图 —— service worker
// 只做两件事：
//   1) 点扩展图标 = 打开侧边栏（setPanelBehavior）
//   2) 页面内悬浮按钮点击 = 打开侧边栏（sidePanel.open 必须在用户手势里调用，
//      内容脚本的 click 经消息转发过来仍然算手势，这是 MV3 官方推荐的写法）

chrome.runtime.onInstalled.addListener(() => {
  chrome.sidePanel
    .setPanelBehavior({ openPanelOnActionClick: true })
    .catch(() => {});
});

chrome.action.onClicked.addListener((tab) => {
  if (!tab || tab.windowId == null) return;
  chrome.sidePanel.open({ windowId: tab.windowId }).catch(() => {});
});

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (!msg || msg.type !== "nfh-open-panel") return false;

  const windowId = sender && sender.tab ? sender.tab.windowId : null;
  const options = windowId != null ? { windowId } : {};

  chrome.sidePanel
    .open(options)
    .then(() => sendResponse({ ok: true }))
    .catch((err) => sendResponse({ ok: false, error: String(err) }));

  return true; // 异步回复
});
