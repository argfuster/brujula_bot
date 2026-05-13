"""
Brújula Trading Bot — EMA + ADX + ATR + Trailing Swing
Modo dual: señal en 1h, entrada confirmada en 15m
=======================================================
Variables Railway:
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, BINANCE_API_KEY, BINANCE_API_SECRET
  USE_TESTNET (true), TRADING_SYMBOL (ETHUSDT), LEVERAGE (3), CAPITAL_PCT (95)
  EMA_PERIOD (25), ADX_PERIOD (14), ADX_MIN (25)
  ATR_PERIOD (14), ATR_MIN_PCT (0.25)
  SL_PCT (0.5), TRAIL_PCT (50), SCAN_INTERVAL (60)

Lógica de entrada:
  1. Señal detectada en vela de 1h (EMA + ADX + ATR)
  2. Busca en velas de 15m la primera que abra en la misma dirección
  3. Máximo 4 velas de 15m (= 1 hora) para confirmar
  4. Si no confirma en 4 velas, descarta y espera nueva señal

Sesiones: NY · Londres · Pre-NY · Asia · Fin de semana (sin Post-NY)
"""

import os
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from binance.client import Client
from binance.enums import SIDE_BUY, SIDE_SELL, ORDER_TYPE_MARKET
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')
BINANCE_KEY      = os.environ.get('BINANCE_API_KEY', '')
BINANCE_SECRET   = os.environ.get('BINANCE_API_SECRET', '')
USE_TESTNET      = os.environ.get('USE_TESTNET', 'true').lower() == 'true'
SYMBOL           = os.environ.get('TRADING_SYMBOL', 'ETHUSDT')
LEVERAGE         = int(os.environ.get('LEVERAGE', '3'))
CAPITAL_PCT      = float(os.environ.get('CAPITAL_PCT', '95'))
EMA_PERIOD       = int(os.environ.get('EMA_PERIOD', '25'))
ADX_PERIOD       = int(os.environ.get('ADX_PERIOD', '14'))
ADX_MIN          = float(os.environ.get('ADX_MIN', '25'))
ATR_PERIOD       = int(os.environ.get('ATR_PERIOD', '14'))
ATR_MIN_PCT      = float(os.environ.get('ATR_MIN_PCT', '0.25'))
SL_PCT           = float(os.environ.get('SL_PCT', '0.5'))
TRAIL_PCT        = float(os.environ.get('TRAIL_PCT', '50'))
SCAN_INTERVAL    = int(os.environ.get('SCAN_INTERVAL', '60'))

TF_SIGNAL = '1h'    # timeframe para señal
TF_ENTRY  = '15m'   # timeframe para confirmación de entrada
MAX_ENTRY_CANDLES = 4  # máximo de velas de 15m para confirmar (= 1 hora)

# ─── ESTADO GLOBAL ────────────────────────────────────────────────────────────
active_trade:       dict | None = None
pending_signal:     str  | None = None   # 'long' o 'short'
pending_signal_ts:  int  | None = None   # timestamp cierre vela de señal (segundos)
pending_signal_close: float | None = None  # close de la vela de señal
last_candle_time:   int  | None = None   # último candle 1h evaluado

# ─── CLIENTE BINANCE ──────────────────────────────────────────────────────────
def get_client() -> Client:
    c = Client(BINANCE_KEY, BINANCE_SECRET, testnet=USE_TESTNET)
    if USE_TESTNET:
        c.FUTURES_URL = 'https://testnet.binancefuture.com/fapi'
    return c

def get_mark_price(symbol: str) -> float:
    try:
        data = get_client().futures_mark_price(symbol=symbol)
        return float(data.get('markPrice', 0))
    except Exception as e:
        log.error(f"Error mark price: {e}")
        return 0.0

def get_klines(symbol: str, interval: str, limit: int = 350) -> pd.DataFrame:
    raw = get_client().futures_klines(symbol=symbol, interval=interval, limit=limit)
    df  = pd.DataFrame(raw, columns=[
        'open_time','open','high','low','close','vol',
        'close_time','qvol','trades','tbb','tbq','ignore'
    ])
    for col in ['open','high','low','close','vol']:
        df[col] = pd.to_numeric(df[col])
    df['open_time'] = df['open_time'].astype(int)
    return df

def get_balance() -> float:
    try:
        for b in get_client().futures_account_balance():
            if b['asset'] == 'USDT':
                return float(b['balance'])
    except Exception as e:
        log.error(f"Error balance: {e}")
    return 0.0

def get_step_size(symbol: str) -> float:
    try:
        info = get_client().futures_exchange_info()
        for s in info['symbols']:
            if s['symbol'] == symbol:
                for f in s['filters']:
                    if f['filterType'] == 'LOT_SIZE':
                        return float(f['stepSize'])
    except Exception as e:
        log.error(f"Error step size: {e}")
    return 0.001

# ─── INDICADORES ──────────────────────────────────────────────────────────────
def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def calc_atr(df: pd.DataFrame, period: int) -> pd.Series:
    hl  = df['high'] - df['low']
    hc  = (df['high'] - df['close'].shift(1)).abs()
    lc  = (df['low']  - df['close'].shift(1)).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def calc_adx(df: pd.DataFrame, period: int) -> pd.Series:
    """ADX con suavizado Wilder (RMA) — alineado con TradingView."""
    up   = df['high'].diff()
    down = -df['low'].diff()
    pdm  = np.where((up > down) & (up > 0), up, 0.0)
    ndm  = np.where((down > up) & (down > 0), down, 0.0)
    pc   = df['close'].shift(1)
    tr   = pd.concat([
        df['high'] - df['low'],
        (df['high'] - pc).abs(),
        (df['low']  - pc).abs()
    ], axis=1).max(axis=1)

    def wilder_smooth(series: pd.Series, n: int) -> pd.Series:
        result = np.full(len(series), np.nan)
        result[n] = series.iloc[1:n+1].sum()
        for i in range(n + 1, len(series)):
            result[i] = result[i-1] - result[i-1] / n + series.iloc[i]
        return pd.Series(result, index=series.index)

    tr_w   = wilder_smooth(tr,                            period)
    pdm_w  = wilder_smooth(pd.Series(pdm, index=df.index), period)
    ndm_w  = wilder_smooth(pd.Series(ndm, index=df.index), period)

    pdi = (pdm_w / tr_w * 100).replace([np.inf, -np.inf], np.nan)
    ndi = (ndm_w / tr_w * 100).replace([np.inf, -np.inf], np.nan)
    dx  = ((pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan) * 100)

    adx = wilder_smooth(dx.fillna(0), period) / period
    return adx

# ─── HORARIO ──────────────────────────────────────────────────────────────────
def in_session(ts_sec: int) -> bool:
    d   = datetime.fromtimestamp(ts_sec, tz=timezone.utc)
    hm  = d.hour * 60 + d.minute
    dow = d.weekday()
    if dow >= 5:                           return True   # finde
    if 13 * 60 + 30 <= hm < 20 * 60:      return True   # NY
    if 8  * 60       <= hm < 12 * 60:     return True   # Londres
    if 12 * 60       <= hm < 13 * 60 + 30: return True  # Pre-NY
    if hm < 8 * 60:                        return True   # Asia
    return False                                         # Post-NY: OFF

# ─── SEÑAL EN 1H ──────────────────────────────────────────────────────────────
def check_signal(df1h: pd.DataFrame) -> tuple[str | None, float, int]:
    """
    Evalúa la penúltima vela de 1h (ya cerrada).
    Retorna (dirección, close_señal, timestamp_cierre) o (None, 0, 0).
    """
    min_bars = max(EMA_PERIOD, ADX_PERIOD * 3, ATR_PERIOD) + 10
    if len(df1h) < min_bars:
        return None, 0, 0

    ema = calc_ema(df1h['close'], EMA_PERIOD)
    atr = calc_atr(df1h, ATR_PERIOD)
    adx = calc_adx(df1h, ADX_PERIOD)

    i     = -2
    close = df1h['close'].iloc[i]
    ema_v = ema.iloc[i]
    atr_v = atr.iloc[i]
    adx_v = adx.iloc[i]
    ts    = int(df1h['open_time'].iloc[i]) // 1000 + 3600  # timestamp de cierre

    if any(pd.isna(x) for x in [ema_v, atr_v, adx_v]) or close <= 0:
        return None, 0, 0
    atr_pct = atr_v / close * 100
    dir_str = 'SHORT' if close < ema_v else ('LONG' if close > ema_v else '=EMA')
    log.info(
        f"Indicadores | close={close:.2f} EMA={ema_v:.2f} "
        f"ADX={adx_v:.2f}/{ADX_MIN} ATR={atr_pct:.3f}%/{ATR_MIN_PCT}% | {dir_str}"
    )

    if adx_v < ADX_MIN:
        log.info(f"Filtrado ADX: {adx_v:.2f} < {ADX_MIN}")
        return None, 0, 0
    if atr_pct < ATR_MIN_PCT:
        log.info(f"Filtrado ATR: {atr_pct:.3f}% < {ATR_MIN_PCT}%")
        return None, 0, 0

    if close > ema_v:  return 'long',  close, ts
    if close < ema_v:  return 'short', close, ts
    return None, 0, 0

# ─── CONFIRMACIÓN EN 15M ──────────────────────────────────────────────────────
def find_entry_15m(direction: str, signal_ts: int, signal_close: float,
                   df15: pd.DataFrame) -> tuple[bool, float, bool]:
    """
    Busca confirmación en velas de 15m desde signal_ts.
    Itera TODAS las velas cerradas disponibles desde la señal.
    Retorna (confirmado, precio_entrada, expirado).
    expirado=True si ya pasaron MAX_ENTRY_CANDLES velas sin confirmar.
    """
    # Velas cerradas posteriores a la señal (excluir la última que puede estar abierta)
    post = df15[(df15['open_time'] // 1000 >= signal_ts)].copy()
    post = post.reset_index(drop=True)

    # Excluir la última vela (puede estar abierta aún)
    if len(post) > 1:
        post = post.iloc[:-1]
    elif len(post) == 1:
        # Solo hay una vela — puede estar aún abierta, esperar
        return False, 0.0, False

    if len(post) == 0:
        return False, 0.0, False

    prev_close = signal_close

    for i in range(len(post)):
        c15    = post.iloc[i]
        open15 = float(c15['open'])

        confirmed = (
            (direction == 'long'  and open15 >= prev_close) or
            (direction == 'short' and open15 <= prev_close)
        )

        if confirmed:
            log.info(f"Confirmación 15m en vela {i+1}/{MAX_ENTRY_CANDLES}: "
                     f"open {open15:.4f} {'≥' if direction=='long' else '≤'} "
                     f"prev_close {prev_close:.4f}")
            return True, open15, False

        prev_close = float(c15['close'])
        log.info(f"Vela 15m {i+1}: no confirma (open {open15:.4f}, prev {prev_close:.4f})")

        # Si llegamos al máximo de velas sin confirmar, expirar
        if i + 1 >= MAX_ENTRY_CANDLES:
            log.info(f"Señal expiró tras {MAX_ENTRY_CANDLES} velas de 15m sin confirmación")
            return False, 0.0, True

    return False, 0.0, False

# ─── GESTIÓN DE POSICIÓN ──────────────────────────────────────────────────────
def check_exit(df1h: pd.DataFrame, trade: dict) -> tuple[bool, str, float]:
    ema   = calc_ema(df1h['close'], EMA_PERIOD)
    i     = -2
    close = df1h['close'].iloc[i]
    ema_v = ema.iloc[i]
    prev_close = df1h['close'].iloc[i - 1]

    direction = trade['direction']
    entry     = trade['entry']
    sl_fixed  = trade['sl_fixed']

    # Actualizar mejor cierre histórico con close de la vela actual
    if direction == 'long' and close > trade['best_swing']:
        trade['best_swing'] = close
    elif direction == 'short' and close < trade['best_swing']:
        trade['best_swing'] = close

    best_swing = trade['best_swing']

    # Calcular trailing
    trail_stop = None
    if direction == 'long':
        swing = best_swing - entry
        if swing > 0:
            trail_stop = entry + swing * (TRAIL_PCT / 100)
    else:
        swing = entry - best_swing
        if swing > 0:
            trail_stop = entry - swing * (TRAIL_PCT / 100)

    # Stop activo = mejor entre SL fijo y trailing
    if direction == 'long':
        active_stop = max(sl_fixed, trail_stop) if trail_stop else sl_fixed
    else:
        active_stop = min(sl_fixed, trail_stop) if trail_stop else sl_fixed

    trade['trail_stop']  = trail_stop
    trade['active_stop'] = active_stop

    # Exit por stop (close de vela 1h)
    if direction == 'long' and close <= active_stop:
        reason = 'trailing' if (trail_stop and trail_stop >= sl_fixed) else 'sl'
        return True, reason, active_stop
    if direction == 'short' and close >= active_stop:
        reason = 'trailing' if (trail_stop and trail_stop <= sl_fixed) else 'sl'
        return True, reason, active_stop

    # Exit por cruce EMA (close de vela 1h)
    if direction == 'long'  and close < ema_v:  return True, 'ema', close
    if direction == 'short' and close > ema_v:  return True, 'ema', close

    return False, '', 0.0

# ─── EJECUCIÓN ────────────────────────────────────────────────────────────────
def open_position(direction: str, entry_price: float | None = None) -> dict | None:
    try:
        client = get_client()

        # Configurar margen cruzado y leverage
        try:
            client.futures_change_margin_type(symbol=SYMBOL, marginType='CROSSED')
        except Exception:
            pass  # Ya está en CROSSED — Binance lanza excepción si no cambia

        lev_resp = client.futures_change_leverage(symbol=SYMBOL, leverage=LEVERAGE)
        log.info(f"Leverage configurado: {lev_resp}")

        balance = get_balance()
        price   = entry_price or get_mark_price(SYMBOL)
        step    = get_step_size(SYMBOL)

        log.info(f"Balance: ${balance:.2f} | Precio: {price:.4f} | Step: {step} | Capital%: {CAPITAL_PCT} | Lev: {LEVERAGE}x")

        if price <= 0 or step <= 0 or balance <= 0:
            log.error(f"Datos inválidos: price={price} step={step} balance={balance}")
            return None

        notional = balance * (CAPITAL_PCT / 100) * LEVERAGE
        qty      = notional / price
        qty      = qty - (qty % step)
        qty      = round(qty, 8)

        if qty <= 0:
            log.error(f"Qty=0 — balance=${balance:.2f} price={price}")
            return None

        side  = SIDE_BUY if direction == 'long' else SIDE_SELL
        order = client.futures_create_order(
            symbol=SYMBOL, side=side,
            type=ORDER_TYPE_MARKET, quantity=qty
        )
        raw_fill = float(order.get('avgPrice') or 0)
        entry    = raw_fill if raw_fill > 0 else price
        sl_fixed = entry * (1 - SL_PCT / 100) if direction == 'long' \
                   else entry * (1 + SL_PCT / 100)

        log.info(f"Abierto: {direction.upper()} {qty} {SYMBOL} @ {entry:.4f} SL={sl_fixed:.4f}")
        return {
            'symbol':      SYMBOL,
            'direction':   direction,
            'qty':         qty,
            'entry':       entry,
            'sl_fixed':    sl_fixed,
            'best_swing':  entry,
            'trail_stop':  None,
            'active_stop': sl_fixed,
            'balance_in':  balance,
            'opened_at':   datetime.now(timezone.utc),
        }
    except Exception as e:
        log.error(f"Error abriendo posición: {e}")
        return None

def close_position(trade: dict) -> float | None:
    try:
        client = get_client()
        side   = SIDE_SELL if trade['direction'] == 'long' else SIDE_BUY
        order  = client.futures_create_order(
            symbol=trade['symbol'], side=side,
            type=ORDER_TYPE_MARKET,
            quantity=trade['qty'], reduceOnly=True
        )
        raw   = float(order.get('avgPrice') or 0)
        price = raw if raw > 0 else get_mark_price(trade['symbol'])
        log.info(f"Cerrado: {trade['symbol']} @ {price:.4f}")
        return price
    except Exception as e:
        log.error(f"Error cerrando: {e}")
        return None

# ─── MENSAJES ─────────────────────────────────────────────────────────────────
def fmt_open(trade: dict) -> str:
    env   = '🧪 TESTNET' if USE_TESTNET else '🔴 REAL'
    emoji = '🟢' if trade['direction'] == 'long' else '🔴'
    return (
        f"{'─'*30}\n⚡ *ENTRADA* {env}\n{'─'*30}\n"
        f"*Par:*      `{trade['symbol']}`\n"
        f"*Dir:*      {emoji} {trade['direction'].upper()}\n"
        f"*Precio:*   `{trade['entry']:,.4f}`\n"
        f"*Cantidad:* `{trade['qty']}`\n"
        f"*SL fijo:*  `{trade['sl_fixed']:,.4f}` (-{SL_PCT}%)\n"
        f"*Trail:*    {TRAIL_PCT}% del swing\n"
        f"*Capital:*  `${trade['balance_in']:,.2f}` × {LEVERAGE}×\n"
        f"{'─'*30}"
    )

def fmt_close(trade: dict, exit_price: float, reason: str) -> str:
    entry    = trade['entry']
    dir_     = trade['direction']
    pnl_pct  = (exit_price - entry) / entry * 100 if dir_ == 'long' \
               else (entry - exit_price) / entry * 100
    pnl_usdt = trade['balance_in'] * (CAPITAL_PCT / 100) * LEVERAGE * pnl_pct / 100
    dur      = datetime.now(timezone.utc) - trade['opened_at']
    h, m     = int(dur.total_seconds()//3600), int((dur.total_seconds()%3600)//60)
    emoji    = '🟢' if dir_ == 'long' else '🔴'
    result   = '✅ WIN' if pnl_pct > 0 else '❌ LOSS'
    reason_str = {'sl':'🛑 Stop Loss fijo','trailing':'📍 Trailing stop',
                  'ema':'🟣 Cruce EMA','manual':'🖐 Cierre manual'}.get(reason, reason)
    trail_str = f"`{trade['trail_stop']:,.4f}`" if trade.get('trail_stop') else '_no activo_'
    return (
        f"{'─'*30}\n🔔 *SALIDA* — {result}\n{'─'*30}\n"
        f"*{trade['symbol']}* {emoji} {dir_.upper()}\n"
        f"*Motivo:*   {reason_str}\n"
        f"*Entrada:*  `{entry:,.4f}`\n"
        f"*Salida:*   `{exit_price:,.4f}`\n"
        f"*SL fijo:*  `{trade['sl_fixed']:,.4f}`\n"
        f"*Trail:*    {trail_str}\n"
        f"*P/L:*      `{pnl_pct:+.3f}%` (`{'+' if pnl_usdt>=0 else ''}{pnl_usdt:.2f} USDT`)\n"
        f"*Duración:* `{h}h {m}m`\n"
        f"{'─'*30}"
    )

async def send_tg(app: Application, text: str) -> None:
    try:
        await app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text, parse_mode='Markdown')
    except Exception as e:
        log.error(f"Telegram error: {e}")

# ─── COMANDOS ─────────────────────────────────────────────────────────────────
async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    global active_trade, pending_signal, pending_signal_ts
    if not active_trade:
        msg = "📭 *Sin posición abierta.*"
        if pending_signal:
            desde = datetime.fromtimestamp(pending_signal_ts, tz=timezone.utc).strftime('%H:%M UTC') if pending_signal_ts else '?'
            msg += f"\n⏳ Señal pendiente: *{pending_signal.upper()}* desde {desde}\n_Buscando confirmación en 15m..._"
        await update.message.reply_text(msg, parse_mode='Markdown')
        return
    try:
        price  = get_mark_price(active_trade['symbol'])
        entry  = active_trade['entry']
        dir_   = active_trade['direction']
        pnl    = (price - entry) / entry * 100 if dir_ == 'long' else (entry - price) / entry * 100
        dur    = datetime.now(timezone.utc) - active_trade['opened_at']
        h, m   = int(dur.total_seconds()//3600), int((dur.total_seconds()%3600)//60)
        trail  = active_trade.get('trail_stop')
        trail_str  = f"`{trail:,.4f}`" if trail else '_aún no activo_'
        active_str = f"`{active_trade.get('active_stop', 0):,.4f}`"
        emoji  = '🟢' if dir_ == 'long' else '🔴'
        msg = (
            f"📊 *Posición activa*\n\n"
            f"*{active_trade['symbol']}* {emoji} {dir_.upper()}\n"
            f"Entrada:     `{entry:,.4f}`\n"
            f"Precio:      `{price:,.4f}`\n"
            f"P/L:         `{pnl:+.3f}%`\n"
            f"SL fijo:     `{active_trade['sl_fixed']:,.4f}`\n"
            f"Trail stop:  {trail_str}\n"
            f"Stop activo: {active_str}\n"
            f"Mejor swing: `{active_trade['best_swing']:,.4f}`\n"
            f"Duración:    `{h}h {m}m`"
        )
    except Exception as e:
        msg = f"⚠️ Error: {e}"
    await update.message.reply_text(msg, parse_mode='Markdown')

async def cmd_close(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    global active_trade
    if not active_trade:
        await update.message.reply_text("📭 Sin posición abierta.")
        return
    exit_price = close_position(active_trade)
    if exit_price:
        msg = fmt_close(active_trade, exit_price, 'manual')
        active_trade = None
        await update.message.reply_text(msg, parse_mode='Markdown')
    else:
        await update.message.reply_text("❌ Error cerrando — verificá en Binance.")

async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    env = '🧪 TESTNET' if USE_TESTNET else '🔴 REAL'
    msg = (
        f"🤖 *Brújula Dual Bot* {env}\n\n"
        f"/status — posición activa y stops\n"
        f"/close  — cerrar manualmente\n"
        f"/help   — este mensaje\n\n"
        f"*Señal:* EMA `{EMA_PERIOD}` · ADX `{ADX_MIN}` · ATR `{ATR_MIN_PCT}%` · `{TF_SIGNAL}`\n"
        f"*Entrada:* primera vela `{TF_ENTRY}` confirmada (máx {MAX_ENTRY_CANDLES})\n"
        f"*Stop:* SL `{SL_PCT}%` fijo + trailing `{TRAIL_PCT}%` del swing\n"
        f"`{SYMBOL}` · `{LEVERAGE}×` · Capital `{CAPITAL_PCT}%`\n"
        f"Sesiones: NY · Londres · Pre-NY · Asia · Finde\n"
        f"Scan cada `{SCAN_INTERVAL}s`"
    )
    await update.message.reply_text(msg, parse_mode='Markdown')

# ─── SCAN ─────────────────────────────────────────────────────────────────────
async def scan_job(app: Application) -> None:
    global active_trade, pending_signal, pending_signal_ts, pending_signal_close, last_candle_time

    # Descargar velas de 1h para señal y gestión de posición
    try:
        df1h = get_klines(SYMBOL, TF_SIGNAL, limit=350)
    except Exception as e:
        log.error(f"Error klines 1h: {e}")
        return

    # Deduplicar por vela de 1h
    current_candle = int(df1h['open_time'].iloc[-2]) // 1000
    if current_candle == last_candle_time:
        # Misma vela 1h — pero si hay señal pendiente, buscar confirmación en 15m
        if pending_signal and not active_trade:
            await _buscar_confirmacion_15m(app)
        else:
            log.info("Vela ya evaluada")
        return
    last_candle_time = current_candle

    # ── GESTIÓN DE POSICIÓN ABIERTA ──────────────────────────────────────────
    if active_trade:
        try:
            should_exit, reason, _ = check_exit(df1h, active_trade)
            if should_exit:
                exit_price = close_position(active_trade)
                if exit_price:
                    msg = fmt_close(active_trade, exit_price, reason)
                    active_trade   = None
                    pending_signal = None
                    pending_signal_ts = None
                    pending_signal_close = None
                    await send_tg(app, msg)
                    log.info(f"Cerrado — motivo: {reason}")
                else:
                    log.error("Fallo en cierre")
            else:
                stop  = active_trade.get('active_stop', 0)
                trail = active_trade.get('trail_stop')
                log.info(
                    f"Trade {active_trade['direction'].upper()} | "
                    f"Stop={stop:.4f}" + (f" Trail={trail:.4f}" if trail else " Trail=pendiente")
                )
        except Exception as e:
            log.error(f"Error check_exit: {e}")
        return

    # ── SEÑAL PENDIENTE — buscar confirmación en 15m ─────────────────────────
    if pending_signal:
        await _buscar_confirmacion_15m(app)
        return

    # ── HORARIO ──────────────────────────────────────────────────────────────
    ts_vela = int(df1h['open_time'].iloc[-2]) // 1000
    if not in_session(ts_vela):
        log.info(f"Fuera de horario ({datetime.fromtimestamp(ts_vela, tz=timezone.utc).strftime('%H:%M UTC')})")
        return

    # ── DETECTAR SEÑAL EN 1H ─────────────────────────────────────────────────
    try:
        signal, sig_close, sig_ts = check_signal(df1h)
    except Exception as e:
        log.error(f"Error check_signal: {e}")
        return

    if signal:
        pending_signal       = signal
        pending_signal_ts    = sig_ts
        pending_signal_close = sig_close
        log.info(f"Señal 1h: {signal.upper()} @ close {sig_close:.4f} — buscando confirmación en 15m")
        await _buscar_confirmacion_15m(app)
    else:
        log.info("Sin señal")

async def _buscar_confirmacion_15m(app: Application) -> None:
    """Busca confirmación de la señal pendiente en velas de 15m."""
    global active_trade, pending_signal, pending_signal_ts, pending_signal_close

    if not pending_signal or not pending_signal_ts:
        return

    try:
        df15 = get_klines(SYMBOL, TF_ENTRY, limit=20)
    except Exception as e:
        log.error(f"Error klines 15m: {e}")
        return

    confirmed, entry_price, expired = find_entry_15m(
        pending_signal, pending_signal_ts, pending_signal_close, df15
    )

    if confirmed:
        signal_dir = pending_signal
        pending_signal       = None
        pending_signal_ts    = None
        pending_signal_close = None
        trade = open_position(signal_dir, entry_price)
        if trade:
            active_trade = trade
            await send_tg(app, fmt_open(trade))
        else:
            log.error("Fallo en apertura")
    elif expired:
        log.info(f"Señal {pending_signal.upper()} descartada — no confirmó en {MAX_ENTRY_CANDLES} velas de 15m")
        pending_signal       = None
        pending_signal_ts    = None
        pending_signal_close = None
    else:
        log.info(f"Sin confirmación 15m aún para señal {pending_signal.upper()}")

# ─── MAIN ─────────────────────────────────────────────────────────────────────
async def scan_callback(ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await scan_job(ctx.application)

async def post_init(app: Application) -> None:
    app.job_queue.run_repeating(
        callback=scan_callback,
        interval=SCAN_INTERVAL,
        first=10,
        name='scan',
    )
    env = 'TESTNET 🧪' if USE_TESTNET else 'REAL 🔴'
    log.info(f"Bot iniciado — {SYMBOL} | Señal:{TF_SIGNAL} Entrada:{TF_ENTRY} | {env} | Lev:{LEVERAGE}x")
    await app.bot.send_message(
        chat_id=TELEGRAM_CHAT_ID,
        parse_mode='Markdown',
        text=(
            f"🤖 *Brújula Dual Bot* {'🧪 TESTNET' if USE_TESTNET else '🔴 REAL'}\n\n"
            f"*Señal:* `{TF_SIGNAL}` — EMA `{EMA_PERIOD}` · ADX `{ADX_MIN}` · ATR `{ATR_MIN_PCT}%`\n"
            f"*Entrada:* primera vela `{TF_ENTRY}` confirmada (máx {MAX_ENTRY_CANDLES})\n"
            f"*Stop:* SL `{SL_PCT}%` fijo + trailing `{TRAIL_PCT}%` del swing\n"
            f"*Par:* `{SYMBOL}` · *Leverage:* `{LEVERAGE}×` · *Capital:* `{CAPITAL_PCT}%`\n"
            f"*Sesiones:* NY · Londres · Pre-NY · Asia · Finde\n\n"
            f"_Scan cada {SCAN_INTERVAL}s. /help para comandos._"
        )
    )

def main() -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        raise ValueError("TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID son requeridos")
    if not BINANCE_KEY or not BINANCE_SECRET:
        raise ValueError("BINANCE_API_KEY y BINANCE_API_SECRET son requeridos")

    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .post_init(post_init)
        .build()
    )
    app.add_handler(CommandHandler('status', cmd_status))
    app.add_handler(CommandHandler('close',  cmd_close))
    app.add_handler(CommandHandler('help',   cmd_help))
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

if __name__ == '__main__':
    main()
