// Hybrid regime-switching BTC bot for make.fun btc-race-24h-20261123-0511.
//
// The bot reads the hello once (capital, caps, fees, phase seed) and then
// answers one batch per minute bar with its orders for that bar. It watches
// the market regime and switches strategy: in a trending regime it rides
// momentum (EMA cross plus a trailing-return filter); in a ranging regime it
// buys dips (oversold RSI, price under a short band) and sells the reversion.
// Long only, one open position, full-size entries, fills at the next bar open.
//
// Measured on the dev window (1,440 one-minute cbBTC bars, BTC +3.64% with a
// slow grind up then a flat second half): the window punishes churning.
// EMA-cross alone loses -$114.83 over 76 trades; buy-and-hold makes +$32.17
// and momentum +$34.94 with a single trade. So this bot trades rarely: one
// confirmed trend entry held behind a wide trailing stop, plus a few dip
// buys in ranges, a cooldown between trades, and a drawdown guard that
// flattens the book and stops trading if equity falls 4% from its peak.
//
// Parameters were measured in an offline fill-model simulator calibrated to
// the official grade (buy-and-hold reproduces +$32.37 vs +$32.17 official)
// and chosen on broad plateaus, not single spikes: trailing stop 2% beats
// 1.5% by ~$26 (the tighter stop gets shaken out mid-push and re-enters
// badly), while 2% / 2.5% / 3% are identical on the window; the trend
// trigger (EMA 8/21, 60-bar return > +0.3%) fires at bar 86, early enough to
// catch the push without chasing it.
//
// Everything here is deterministic: float64 arithmetic on the closes, no
// randomness (nothing is seeded from hello.seed because nothing is random),
// no wall clock, no environment, no files outside /bot.

import { runBars, type Bar, type BarsState, type BuyInput, type SellInput } from './vendor/makefun-bars.ts';

// --- regime + indicator settings (measured, see header) -------------------
const FAST_SPAN = 8;
const SLOW_SPAN = 21;
const RSI_SPAN = 14;
const BAND_SPAN = 20;
const BAND_SIGMA = 2;
// Trend entry: fast EMA above slow EMA, close above fast EMA, 60-bar return
// above +0.3%, 30-bar return positive.
const TREND_RET60 = 0.003;
// Dip entry: RSI under 30, close at or under the lower band, and the 60-bar
// return between -1% (not a waterfall) and +0.2% (not an uptrend pullback).
const DIP_RSI = 30;
const DIP_RET60_MAX = 0.002;
const DIP_RET60_MIN = -0.01;
// Trend exit: 2% trailing stop from the peak close, 2.5% hard stop.
const TRAIL_TREND = 0.02;
const STOP_TREND = 0.025;
// Range exit: +0.9% take-profit, 1% trailing stop, 1.5% hard stop, or a
// reversion to the band middle once the trade is +0.3% or better.
const TAKE_RANGE = 0.009;
const TRAIL_RANGE = 0.01;
const STOP_RANGE = 0.015;
const REVERT_MIN_GAIN = 0.003;
// Risk: 60 bars of cooldown after any sell; flatten + stop if equity falls
// 4% from its peak.
const COOLDOWN_BARS = 60;
const DD_GUARD_BPS = 400;
const MAX_SLIPPAGE_BPS = 100;

type Mode = 'trend' | 'range';

interface Ema {
  span: number;
  warm: number[];
  value: number | null;
}

interface Rsi {
  span: number;
  seen: number;
  gain: number;
  loss: number;
  value: number | null;
}

interface Series {
  closes: number[];
  fast: Ema;
  slow: Ema;
  rsi: Rsi;
  bandLo: number | null;
  bandMid: number | null;
  // Current trade, set when a buy is decided, cleared when flat.
  mode: Mode | null;
  entry: number;
  peak: number;
  lastSellSeq: number;
}

const seriesBySymbol = new Map<string, Series>();
let peakEquity = 0n;
let stopped = false;

function newSeries(): Series {
  return {
    closes: [],
    fast: { span: FAST_SPAN, warm: [], value: null },
    slow: { span: SLOW_SPAN, warm: [], value: null },
    rsi: { span: RSI_SPAN, seen: 0, gain: 0, loss: 0, value: null },
    bandLo: null,
    bandMid: null,
    mode: null,
    entry: 0,
    peak: 0,
    lastSellSeq: -1_000_000,
  };
}

function emaStep(e: Ema, close: number): void {
  if (e.value !== null) {
    e.value += (2 / (e.span + 1)) * (close - e.value);
    return;
  }
  e.warm.push(close);
  if (e.warm.length === e.span) {
    e.value = e.warm.reduce((a, x) => a + x, 0) / e.span;
  }
}

function rsiStep(r: Rsi, closes: number[]): void {
  const n = closes.length;
  if (n < 2) return;
  const d = closes[n - 1] - closes[n - 2];
  if (r.seen < r.span) {
    // Seed with the simple average gain/loss of the first span of moves.
    if (d > 0) r.gain += d;
    else r.loss -= d;
    r.seen += 1;
    if (r.seen === r.span) {
      r.gain /= r.span;
      r.loss /= r.span;
      r.value = r.loss === 0 ? 100 : 100 - 100 / (1 + r.gain / r.loss);
    }
    return;
  }
  // Wilder smoothing after the seed.
  const up = d > 0 ? d : 0;
  const dn = d < 0 ? -d : 0;
  r.gain = (r.gain * (r.span - 1) + up) / r.span;
  r.loss = (r.loss * (r.span - 1) + dn) / r.span;
  r.value = r.loss === 0 ? 100 : 100 - 100 / (1 + r.gain / r.loss);
}

function bandStep(s: Series): void {
  const n = s.closes.length;
  if (n < BAND_SPAN) {
    s.bandLo = null;
    s.bandMid = null;
    return;
  }
  const w = s.closes.slice(n - BAND_SPAN);
  const mean = w.reduce((a, x) => a + x, 0) / BAND_SPAN;
  const variance = w.reduce((a, x) => a + (x - mean) * (x - mean), 0) / BAND_SPAN;
  s.bandMid = mean;
  s.bandLo = mean - BAND_SIGMA * Math.sqrt(variance);
}

/** Fold one bar into its symbol's indicators. A missing bar adds nothing. */
function pushBar(bar: Bar): Series {
  let s = seriesBySymbol.get(bar.symbol);
  if (s === undefined) {
    s = newSeries();
    seriesBySymbol.set(bar.symbol, s);
  }
  const close = Number(bar.c);
  s.closes.push(close);
  emaStep(s.fast, close);
  emaStep(s.slow, close);
  rsiStep(s.rsi, s.closes);
  bandStep(s);
  return s;
}

/** Trailing return over `n` bars ending at the latest close, or null. */
function trailingReturn(s: Series, n: number): number | null {
  const len = s.closes.length;
  if (len <= n) return null;
  return s.closes[len - 1] / s.closes[len - 1 - n] - 1;
}

function sellAll(symbol: string, qty: bigint): SellInput {
  return { side: 'sell', symbol, qty, max_slippage_bps: MAX_SLIPPAGE_BPS };
}

/** One all-in buy of the symbol: all the cash, capped by the weight room. */
function buyFull(state: BarsState, symbol: string): BuyInput | null {
  const room = state.weightRoom(symbol);
  const slice = state.cash < room ? state.cash : room;
  if (slice === 0n || slice < BigInt(state.hello.fees.min_notional_quote_units)) return null;
  return { side: 'buy', symbol, notional_quote_units: slice, max_slippage_bps: MAX_SLIPPAGE_BPS };
}

await runBars({
  onBatch(state: BarsState, batch) {
    for (const bar of batch.bars) pushBar(bar);
    // Warm-up bars seed the indicators; the replay rejects every order
    // decided on one, so send none.
    if (state.warmingUp) return [];

    // Drawdown guard on peak equity (marked at the latest closes).
    if (state.equity > peakEquity) peakEquity = state.equity;
    if (!stopped && peakEquity > 0n && (peakEquity - state.equity) * 10_000n > peakEquity * BigInt(DD_GUARD_BPS)) {
      stopped = true;
    }

    const orders: (BuyInput | SellInput)[] = [];
    for (const { symbol } of state.hello.symbols) {
      const s = seriesBySymbol.get(symbol);
      if (s === undefined) continue;
      const held = state.positions.get(symbol);
      const close = Number(state.close(symbol) ?? 0n);
      if (close <= 0) continue;

      if (held !== undefined) {
        // Defensive: a position with no trade memory (cannot happen in one
        // run) falls back to trend exits priced off its cost.
        if (s.mode === null) {
          s.mode = 'trend';
          s.entry = Number((held.cost * 10_000_000n) / (held.qty === 0n ? 1n : held.qty));
          s.peak = s.entry;
        }
        if (close > s.peak) s.peak = close;
        let exit = false;
        if (s.mode === 'trend') {
          if (close < s.peak * (1 - TRAIL_TREND)) exit = true;
          if (close < s.entry * (1 - STOP_TREND)) exit = true;
        } else {
          if (close >= s.entry * (1 + TAKE_RANGE)) exit = true;
          if (close < s.peak * (1 - TRAIL_RANGE)) exit = true;
          if (close < s.entry * (1 - STOP_RANGE)) exit = true;
          if (s.bandMid !== null && close >= s.bandMid && close > s.entry * (1 + REVERT_MIN_GAIN)) exit = true;
        }
        if (stopped) exit = true;
        if (exit) {
          // Keep s.mode set: if the market refuses this sell (max
          // slippage), the next bar retries it under the same mode; the
          // flat branch clears the memory once the position is gone.
          orders.push(sellAll(symbol, held.qty));
          s.lastSellSeq = state.seq;
        }
        continue;
      }

      // Flat: clear any stale trade memory and look for an entry.
      s.mode = null;
      if (stopped) continue;
      if (state.seq - s.lastSellSeq < COOLDOWN_BARS) continue;
      if (s.fast.value === null || s.slow.value === null) continue;
      const ret30 = trailingReturn(s, 30);
      const ret60 = trailingReturn(s, 60);
      if (ret30 === null || ret60 === null) continue;

      const trending =
        s.fast.value > s.slow.value && close > s.fast.value && ret60 > TREND_RET60 && ret30 > 0;
      if (trending) {
        const buy = buyFull(state, symbol);
        if (buy === null) continue;
        orders.push(buy);
        s.mode = 'trend';
        s.entry = close;
        s.peak = close;
        continue;
      }

      const oversold =
        s.rsi.value !== null &&
        s.rsi.value < DIP_RSI &&
        s.bandLo !== null &&
        close <= s.bandLo &&
        ret60 < DIP_RET60_MAX &&
        ret60 > DIP_RET60_MIN;
      if (oversold) {
        const buy = buyFull(state, symbol);
        if (buy === null) continue;
        orders.push(buy);
        s.mode = 'range';
        s.entry = close;
        s.peak = close;
      }
    }
    return orders;
  },
});
