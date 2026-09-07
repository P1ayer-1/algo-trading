// Live chart page logic: connects to the backend's websocket, keeps a small
// local `state` object in sync, and renders it into the lightweight-charts
// candlestick chart + the supports/resistances tables.
//
// Sections below: state, chart setup, rendering, websocket connection.

// --- Shared page state -----------------------------------------------------
const state = {
  candles: [],
  supports: [],
  resistances: [],
  currentPrice: null,
  instId: "BTC-USDT",
  bar: "15m",
  updatedAt: null,
  features: null,
  feedStatus: null
};

const chartEl = document.getElementById("chart");
const statusEl = document.getElementById("status");
const titleEl = document.getElementById("title");
const subtitleEl = document.getElementById("subtitle");
const lastPriceEl = document.getElementById("lastPrice");
const supportsEl = document.getElementById("supports");
const resistancesEl = document.getElementById("resistances");
const feedStateEl = document.getElementById("feedState");
const signalsEl = document.getElementById("signals");
const flowEl = document.getElementById("flow");

// --- Chart setup -------------------------------------------------------------
const styles = getComputedStyle(document.documentElement);
const colors = {
  bg: styles.getPropertyValue("--panel").trim(),
  text: styles.getPropertyValue("--text").trim(),
  muted: styles.getPropertyValue("--muted").trim(),
  border: styles.getPropertyValue("--border").trim(),
  support: styles.getPropertyValue("--support").trim(),
  resistance: styles.getPropertyValue("--resistance").trim(),
  current: styles.getPropertyValue("--current").trim()
};

const chart = LightweightCharts.createChart(chartEl, {
  autoSize: true,
  layout: {
    background: { color: colors.bg },
    textColor: colors.muted,
    attributionLogo: false
  },
  grid: {
    vertLines: { color: colors.border },
    horzLines: { color: colors.border }
  },
  crosshair: {
    mode: LightweightCharts.CrosshairMode.Normal
  },
  rightPriceScale: {
    borderColor: colors.border,
    scaleMargins: { top: 0.08, bottom: 0.12 }
  },
  timeScale: {
    borderColor: colors.border,
    timeVisible: true,
    secondsVisible: false
  },
  localization: {
    priceFormatter: formatPrice
  }
});

const candleSeries = chart.addSeries(LightweightCharts.CandlestickSeries, {
  upColor: colors.support,
  downColor: colors.resistance,
  borderUpColor: colors.support,
  borderDownColor: colors.resistance,
  wickUpColor: colors.support,
  wickDownColor: colors.resistance
});

const priceLines = {
  supports: [],
  resistances: [],
  current: null
};
let loadedRangeKey = "";

// --- Formatting / conversion helpers -----------------------------------------
function formatPrice(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return Number(value).toLocaleString(undefined, {
    minimumFractionDigits: 1,
    maximumFractionDigits: 1
  });
}

function toChartCandles(candles) {
  return candles.map(toChartCandle);
}

function toChartCandle(candle) {
  return {
    time: Math.floor(candle.ts / 1000),
    open: candle.open,
    high: candle.high,
    low: candle.low,
    close: candle.close
  };
}

// --- Rendering ----------------------------------------------------------------
function removeLines(lines) {
  for (const line of lines) {
    candleSeries.removePriceLine(line);
  }
}

function setPriceLines() {
  removeLines(priceLines.supports);
  removeLines(priceLines.resistances);
  if (priceLines.current) candleSeries.removePriceLine(priceLines.current);

  priceLines.supports = state.supports.map((level, index) => candleSeries.createPriceLine({
    price: level.level,
    color: colors.support,
    lineWidth: 1,
    lineStyle: LightweightCharts.LineStyle.Solid,
    axisLabelVisible: true,
    title: `S${index + 1} ${level.touches}x`
  }));

  priceLines.resistances = state.resistances.map((level, index) => candleSeries.createPriceLine({
    price: level.level,
    color: colors.resistance,
    lineWidth: 1,
    lineStyle: LightweightCharts.LineStyle.Solid,
    axisLabelVisible: true,
    title: `R${index + 1} ${level.touches}x`
  }));

  priceLines.current = state.currentPrice === null ? null : candleSeries.createPriceLine({
    price: state.currentPrice,
    color: colors.current,
    lineWidth: 2,
    lineStyle: LightweightCharts.LineStyle.Dashed,
    axisLabelVisible: true,
    title: "last"
  });
}

function renderChart() {
  if (!state.candles.length) return;

  const first = state.candles[0];
  const last = state.candles[state.candles.length - 1];
  const rangeKey = `${state.candles.length}:${first.ts}:${last.ts}`;

  if (rangeKey !== loadedRangeKey) {
    candleSeries.setData(toChartCandles(state.candles));
    if (!loadedRangeKey) chart.timeScale().fitContent();
    loadedRangeKey = rangeKey;
  } else {
    candleSeries.update(toChartCandle(last));
  }

  setPriceLines();
}

function renderTable(target, levels) {
  target.innerHTML = "";
  for (const level of levels) {
    const tr = document.createElement("tr");
    const distance = state.currentPrice === null ? null : Math.abs(level.level - state.currentPrice);
    tr.innerHTML = `<td>${formatPrice(level.level)}</td><td>${level.touches}</td><td>${formatPrice(distance)}</td>`;
    target.appendChild(tr);
  }
  if (!levels.length) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="3">None in range</td>`;
    target.appendChild(tr);
  }
}

// --- Microstructure panel ------------------------------------------------------
// Renders the OBI/OFI/trade-flow signals coming from backend/trading. All of
// these are *descriptive*: they show what the book and tape are doing. None of
// them is a trade instruction, and nothing here places orders.

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return Number(value).toFixed(digits);
}

// Signed values in [-1, 1] drive a bar that grows out from the centre.
function signalBar(label, value, formatted) {
  const clamped = Math.max(-1, Math.min(1, Number(value) || 0));
  const width = Math.abs(clamped) * 50;
  const left = clamped >= 0 ? 50 : 50 - width;
  const negative = clamped < 0 ? " negative" : "";
  return `
    <div class="signal">
      <div class="signal-head">
        <span class="label">${label}</span>
        <span class="value">${formatted}</span>
      </div>
      <div class="bar">
        <div class="fill${negative}" style="left:${left}%;width:${width}%"></div>
      </div>
    </div>`;
}

function statRow(label, value) {
  return `<div class="stat-row"><span class="label">${label}</span><span>${value}</span></div>`;
}

function renderFeedState() {
  const status = state.feedStatus;
  const features = state.features;

  if (!status) {
    feedStateEl.className = "feed-state";
    feedStateEl.textContent = "feed offline";
    return false;
  }
  if (!status.connected) {
    feedStateEl.className = "feed-state bad";
    feedStateEl.textContent = `disconnected - ${status.reconnects || 0} reconnects`;
    return false;
  }
  if (!status.bookReady) {
    feedStateEl.className = "feed-state warn";
    feedStateEl.textContent = "resyncing order book...";
    return false;
  }
  if (!features || !features.is_valid) {
    feedStateEl.className = "feed-state warn";
    feedStateEl.textContent = "waiting for a valid book snapshot";
    return false;
  }

  const levels = status.bookLevels || {};
  const recorder = status.recorder || {};
  const recorded = recorder.enabled ? `${recorder.rowsWritten || 0} rows` : "off";
  feedStateEl.className = "feed-state ok";
  feedStateEl.textContent =
    `live - ${levels.bids || 0}x${levels.asks || 0} levels, ` +
    `${status.resyncs || 0} resyncs, recording ${recorded}`;
  return true;
}

function renderSignals() {
  const healthy = renderFeedState();
  const features = state.features;

  if (!healthy || !features) {
    signalsEl.innerHTML = "";
    flowEl.innerHTML = `<tr><td colspan="3">No feed data</td></tr>`;
    return;
  }

  // OFI is unbounded (it is a size, not a ratio), so scale it against recent
  // depth to get something bar-shaped. This is display normalisation only —
  // the raw value is what gets recorded and modelled.
  const depth = (features.bid_depth_20 || 0) + (features.ask_depth_20 || 0);
  const ofiScaled = depth > 0 ? features.ofi_5s / (depth / 2) : 0;

  signalsEl.innerHTML = [
    signalBar("Book imbalance (top)", features.obi_1, formatNumber(features.obi_1)),
    signalBar("Book imbalance (20)", features.obi_20, formatNumber(features.obi_20)),
    signalBar("Order flow 5s", ofiScaled, formatNumber(features.ofi_5s, 1)),
    signalBar("Trade flow 5s", features.tfi_5s, formatNumber(features.tfi_5s)),
    statRow("Microprice", formatPrice(features.microprice)),
    statRow("Micro - mid", `${formatNumber(features.microprice_delta_bps, 2)} bps`),
    statRow("Spread", `${formatNumber(features.spread_bps, 2)} bps`),
    statRow("Realised vol 60s", formatNumber(features.rv_60s * 10000, 2)),
    statRow(
      "Regime",
      `<span class="regime ${features.vol_regime}">${features.vol_regime.replace("_", " ")}</span>`
    ),
    statRow("Funding", `${formatNumber((features.funding_rate || 0) * 100, 4)}%`)
  ].join("");

  const rows = [
    ["Trade flow", features.tfi_1s, features.tfi_5s],
    ["Order flow", features.ofi_1s, features.ofi_5s],
    ["Return (bps)", features.ret_1s, features.ret_5s]
  ];
  flowEl.innerHTML = rows
    .map(
      ([label, one, five]) =>
        `<tr><td>${label}</td><td>${formatNumber(one)}</td><td>${formatNumber(five)}</td></tr>`
    )
    .join("");
}

function applyUpdate(payload) {
  state.candles = payload.candles || state.candles;
  state.supports = payload.supports || state.supports;
  state.resistances = payload.resistances || state.resistances;
  state.currentPrice = payload.currentPrice ?? state.currentPrice;
  state.instId = payload.instId || state.instId;
  state.bar = payload.bar || state.bar;
  state.updatedAt = payload.updatedAt || state.updatedAt;
  // `?? null` rather than `|| state.features`: an explicit null from the
  // backend means "feed is not running", which must not be masked by the last
  // good value we happen to be holding.
  state.features = payload.features ?? null;
  state.feedStatus = payload.feedStatus ?? null;

  titleEl.textContent = `${state.instId} live chart`;
  subtitleEl.textContent = `${state.bar} candles, support and resistance recalculated from live/demo BloFin data`;
  lastPriceEl.textContent = formatPrice(state.currentPrice);
  renderTable(supportsEl, state.supports);
  renderTable(resistancesEl, state.resistances);
  renderSignals();
  renderChart();
}

// --- Websocket connection (with auto-reconnect) --------------------------------
let socket = null;
let reconnectTimer = null;
let shouldReconnect = true;

function connect() {
  if (reconnectTimer) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }

  if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {
    return;
  }

  socket = new WebSocket(`ws://${location.hostname || "127.0.0.1"}:8766`);
  statusEl.textContent = "connecting";

  socket.addEventListener("open", () => {
    statusEl.textContent = "live";
  });

  socket.addEventListener("message", (event) => {
    applyUpdate(JSON.parse(event.data));
  });

  socket.addEventListener("close", () => {
    socket = null;
    if (!shouldReconnect) return;
    statusEl.textContent = "reconnecting";
    reconnectTimer = setTimeout(connect, 1500);
  });

  socket.addEventListener("error", () => {
    statusEl.textContent = "connection error";
  });
}

window.addEventListener("beforeunload", () => {
  shouldReconnect = false;
  if (reconnectTimer) clearTimeout(reconnectTimer);
  if (socket) socket.close();
});

connect();
