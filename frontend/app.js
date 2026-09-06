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
  updatedAt: null
};

const chartEl = document.getElementById("chart");
const statusEl = document.getElementById("status");
const titleEl = document.getElementById("title");
const subtitleEl = document.getElementById("subtitle");
const lastPriceEl = document.getElementById("lastPrice");
const supportsEl = document.getElementById("supports");
const resistancesEl = document.getElementById("resistances");

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

function applyUpdate(payload) {
  state.candles = payload.candles || state.candles;
  state.supports = payload.supports || state.supports;
  state.resistances = payload.resistances || state.resistances;
  state.currentPrice = payload.currentPrice ?? state.currentPrice;
  state.instId = payload.instId || state.instId;
  state.bar = payload.bar || state.bar;
  state.updatedAt = payload.updatedAt || state.updatedAt;

  titleEl.textContent = `${state.instId} live chart`;
  subtitleEl.textContent = `${state.bar} candles, support and resistance recalculated from live/demo BloFin data`;
  lastPriceEl.textContent = formatPrice(state.currentPrice);
  renderTable(supportsEl, state.supports);
  renderTable(resistancesEl, state.resistances);
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
