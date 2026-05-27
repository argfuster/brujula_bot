"""
Brújula Bot EMA15m — EMA200 + ADX30 + Trailing 80%
====================================================
Modelo validado contra tester EMA15m (2020-2025, 0 trimestres negativos):
  Señal:    15m — vela anterior cierra > EMA200 → LONG, < EMA200 → SHORT
  Filtro:   ADX14 >= 30
  Entrada:  al OPEN de la vela siguiente a la señal
  SL fijo:  2.5% desde la entrada → STOP_MARKET inmediata en Binance (Algo Orders)
  Trailing: swing crece con CLOSE de velas de 15m ya cerradas desde la entrada
            trail = entrada + swing × 80% (long) / entrada - swing × 80% (short)
            cuando trail supera SL fijo → reemplaza la STOP_MARKET
  Salida:   solo por STOP_MARKET (SL fijo o trailing) — nunca por corte de EMA
  Reentrada: DESACTIVADA — una entrada por vela 15m, espera próxima señal

Variables Railway:
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
  BINANCE_API_KEY, BINANCE_API_SECRET
  USE_TESTNET       (true)
  LEVERAGE          (3)
  SL_PCT            (2.5)
  TRAIL_PCT         (80)
  SCAN_INTERVAL     (30)   ← cada 30s para detectar nuevas velas de 15m
"""

import os, time, logging, math, hmac, hashlib, urllib.parse
import requests as req_lib
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
SYMBOL           = 'ETHUSDT'
LEVERAGE         = int(os.environ.get('LEVERAGE', '3'))
CAPITAL_PCT      = 100.0
EMA_PERIOD       = 200
ADX_PERIOD       = 14
ADX_MIN          = 30.0
SL_PCT           = float(os.environ.get('SL_PCT', '2.5'))
TRAIL_PCT        = float(os.environ.get('TRAIL_PCT', '80'))
SCAN_INTERVAL    = int(os.environ.get('SCAN_INTERVAL', '30'))

# ─── ESTADO GLOBAL ────────────────────────────────────────────────────────────
active_trade:      dict | None = None
last_15m_candle:   int  | None = None  # open_time de la última vela 15m evaluada

# ─── CLIENTE BINANCE ──────────────────────────────────────────────────────────
def get_client() -> Client:
    c = Client(BINANCE_KEY, BINANCE_SECRET, testnet=USE_TESTNET)
    if USE_TESTNET:
        c.FUTURES_URL = 'https://testnet.binancefuture.com/fapi'
    return c

def get_klines(symbol: str, interval: str, limit: int = 300) -> pd.DataFrame:
    raw = get_client().futures_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(raw, columns=[
        'open_time','open','high','low','close','volume',
        'close_time','qav','trades','tbbav','tbqav','ignore'
    ])
    for col in ['open','high','low','close']:
        df[col] = df[col].astype(float)
    df['open_time'] = df['open_time'].astype(int)
    return df

def get_mark_price(symbol: str) -> float:
    """Lanza excepción si falla — nunca retornar 0 para evitar falsos stops."""
    return float(get_client().futures_mark_price(symbol=symbol)['markPrice'])

def get_balance() -> float:
    try:
        for b in get_client().futures_account_balance():
            if b['asset'] == 'USDT':
                return float(b['availableBalance'])
    except Exception as e:
        log.error(f"Error balance: {e}")
    return 0.0

def get_step_size(symbol: str) -> float:
    info = get_client().futures_exchange_info()
    for s in info['symbols']:
        if s['symbol'] == symbol:
            for f in s['filters']:
                if f['filterType'] == 'LOT_SIZE':
                    return float(f['stepSize'])
    return 0.001

def round_qty(qty: float, step: float) -> float:
    if step <= 0: return qty
    decimals = max(0, round(-math.log10(step)))
    return round(math.floor(qty / step) * step, decimals)

# ─── INDICADORES ──────────────────────────────────────────────────────────────
def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def calc_adx(df: pd.DataFrame, period: int) -> pd.Series:
    up   = df['high'].diff()
    down = -df['low'].diff()
    pdm  = np.where((up > down) & (up > 0), up, 0.0)
    ndm  = np.where((down > up) & (down > 0), down, 0.0)
    pc   = df['close'].shift(1)
    tr   = pd.concat([df['high']-df['low'],
                      (df['high']-pc).abs(),
                      (df['low']-pc).abs()], axis=1).max(axis=1)

    def wilder(s, n):
        r = np.full(len(s), np.nan)
        arr = s.values
        r[n] = arr[1:n+1].sum()
        for i in range(n+1, len(arr)):
            r[i] = r[i-1] - r[i-1]/n + arr[i]
        return pd.Series(r, index=s.index)

    tr_w  = wilder(tr, period)
    pdm_w = wilder(pd.Series(pdm, index=df.index), period)
    ndm_w = wilder(pd.Series(ndm, index=df.index), period)
    pdi   = (pdm_w / tr_w * 100).replace([np.inf, -np.inf], np.nan)
    ndi   = (ndm_w / tr_w * 100).replace([np.inf, -np.inf], np.nan)
    dx    = ((pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan) * 100)
    return wilder(dx.fillna(0), period) / period

# ─── ÓRDENES BINANCE ──────────────────────────────────────────────────────────
def place_stop_order(symbol: str, direction: str, qty: float, stop_price: float) -> str | None:
    """Coloca STOP_MARKET via Algo Orders endpoint (POST /fapi/v1/algoOrder)."""
    if USE_TESTNET:
        log.info(f"Testnet: stop software configurado {direction.upper()} stop={stop_price:.4f}")
        return None
    try:
        base_url = "https://fapi.binance.com"
        side = "SELL" if direction == "long" else "BUY"
        ts = int(time.time() * 1000)
        params = {
            "symbol":       symbol,
            "side":         side,
            "type":         "STOP_MARKET",
            "algoType":     "CONDITIONAL",
            "quantity":     str(qty),
            "triggerPrice": f"{stop_price:.2f}",
            "reduceOnly":   "true",
            "workingType":  "MARK_PRICE",
            "timestamp":    ts,
        }
        query = urllib.parse.urlencode(params)
        sig   = hmac.new(BINANCE_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
        params["signature"] = sig
        headers = {"X-MBX-APIKEY": BINANCE_KEY}
        r = req_lib.post(f"{base_url}/fapi/v1/algoOrder", params=params, headers=headers, timeout=10)
        data = r.json()
        if "algoId" in data:
            algo_id = str(data["algoId"])
            log.info(f"STOP_MARKET algo colocada: {direction.upper()} stop={stop_price:.4f} algoId={algo_id}")
            return algo_id
        else:
            log.error(f"Error colocando algo stop: {data} — fallback a stop software")
            return None
    except Exception as e:
        log.error(f"Error colocando STOP_MARKET algo: {e} — fallback a stop software")
        return None

def cancel_stop_order(symbol: str, order_id: str | None) -> bool:
    if not order_id:
        return True
    if USE_TESTNET:
        return True
    try:
        base_url = "https://fapi.binance.com"
        ts = int(time.time() * 1000)
        params = {"algoId": int(order_id), "timestamp": ts}
        query = urllib.parse.urlencode(params)
        sig   = hmac.new(BINANCE_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
        params["signature"] = sig
        headers = {"X-MBX-APIKEY": BINANCE_KEY}
        r = req_lib.delete(f"{base_url}/fapi/v1/algoOrder", params=params, headers=headers, timeout=10)
        data = r.json()
        if data.get("algoId") or data.get("code") == 200:
            log.info(f"Algo order {order_id} cancelada")
            return True
        get_client().futures_cancel_order(symbol=symbol, orderId=int(order_id))
        return True
    except Exception as e:
        log.warning(f"Cancel {order_id}: {e}")
        return False

def open_position(symbol: str, direction: str) -> dict | None:
    """Abre posición MARKET y coloca STOP_MARKET inmediata."""
    try:
        balance  = get_balance()
        mark     = get_mark_price(symbol)
        step     = get_step_size(symbol)
        notional = balance * (CAPITAL_PCT / 100) * LEVERAGE
        qty      = round_qty(notional / mark, step)

        if qty <= 0:
            log.error("Qty calculada = 0, no abre")
            return None

        # Configurar leverage
        get_client().futures_change_leverage(symbol=symbol, leverage=LEVERAGE)

        # Orden MARKET
        side  = SIDE_BUY if direction == 'long' else SIDE_SELL
        order = get_client().futures_create_order(
            symbol=symbol, side=side, type=ORDER_TYPE_MARKET, quantity=qty
        )

        # Precio de entrada real
        fills      = order.get('fills', [])
        entry_price = float(order.get('avgPrice', 0))
        if not entry_price and fills:
            total_qty = sum(float(f['qty']) for f in fills)
            entry_price = sum(float(f['price']) * float(f['qty']) for f in fills) / total_qty if total_qty else mark
        if not entry_price:
            entry_price = mark

        # SL fijo
        sl_fixed = entry_price * (1 - SL_PCT/100) if direction == 'long' else entry_price * (1 + SL_PCT/100)
        stop_id  = place_stop_order(symbol, direction, qty, sl_fixed)

        return {
            'symbol':        symbol,
            'direction':     direction,
            'qty':           qty,
            'entry':         entry_price,
            'sl_fixed':      sl_fixed,
            'best_swing':    entry_price,  # swing parte del precio de entrada
            'trail_stop':    None,
            'stop_order_id': stop_id,
            'opened_at':     datetime.now(timezone.utc),
        }
    except Exception as e:
        log.error(f"Error abriendo posición: {e}")
        return None

def close_position_market(trade: dict) -> float | None:
    """Cierra posición con orden MARKET. Retorna precio de ejecución."""
    try:
        cancel_stop_order(trade['symbol'], trade.get('stop_order_id'))
        side  = SIDE_SELL if trade['direction'] == 'long' else SIDE_BUY
        order = get_client().futures_create_order(
            symbol=trade['symbol'], side=side,
            type=ORDER_TYPE_MARKET, quantity=trade['qty'],
            reduceOnly=True
        )
        fills = order.get('fills', [])
        if fills:
            total_qty = sum(float(f['qty']) for f in fills)
            return sum(float(f['price'])*float(f['qty']) for f in fills) / total_qty if total_qty else None
        avg = float(order.get('avgPrice', 0))
        return avg if avg else None
    except Exception as e:
        log.error(f"Error cerrando posición: {e}")
        return None

# ─── SEÑAL 15M ────────────────────────────────────────────────────────────────
def check_signal_15m(df: pd.DataFrame) -> tuple[str | None, float | None]:
    """
    Evalúa la penúltima vela cerrada (iloc[-2]).
    Retorna (direction, entry_price) o (None, None).
    entry_price = open de la última vela (vela siguiente a la señal).
    """
    if len(df) < EMA_PERIOD + ADX_PERIOD * 3:
        return None, None

    ema = calc_ema(df['close'], EMA_PERIOD)
    adx = calc_adx(df, ADX_PERIOD)

    # Penúltima vela = última cerrada
    close_prev = float(df['close'].iloc[-2])
    ema_prev   = float(ema.iloc[-2])
    adx_prev   = float(adx.iloc[-2])

    if pd.isna(adx_prev) or pd.isna(ema_prev):
        return None, None

    # Filtro ADX
    if adx_prev < ADX_MIN:
        return None, None

    # Señal
    entry_price = float(df['open'].iloc[-1])  # open de la vela actual (siguiente a la señal)

    if close_prev > ema_prev:
        return 'long', entry_price
    elif close_prev < ema_prev:
        return 'short', entry_price
    return None, None

# ─── TRAILING 15M ─────────────────────────────────────────────────────────────
def update_trail_15m(trade: dict, df15: pd.DataFrame) -> bool:
    """
    Actualiza el trailing con cierres de velas 15m favorables desde la entrada.
    Retorna True si el trail mejoró y la STOP_MARKET fue actualizada.
    """
    opened_at_ts = trade['opened_at'].timestamp()
    direction    = trade['direction']
    entry        = trade['entry']
    best_swing   = trade['best_swing']

    # Velas 15m que cerraron DESPUÉS de la apertura
    recent = df15[df15['open_time'] // 1000 + 900 > opened_at_ts]  # close_time > opened_at

    updated = False
    for _, row in recent.iterrows():
        close = float(row['close'])
        if direction == 'long'  and close > best_swing:
            best_swing = close
            updated = True
        elif direction == 'short' and close < best_swing:
            best_swing = close
            updated = True

    if not updated:
        return False

    # Calcular nuevo trail
    if direction == 'long':
        swing = best_swing - entry
        new_trail = entry + swing * (TRAIL_PCT / 100) if swing > 0 else None
    else:
        swing = entry - best_swing
        new_trail = entry - swing * (TRAIL_PCT / 100) if swing > 0 else None

    if new_trail is None:
        return False

    # Trail solo toma control cuando supera al SL fijo
    sl_fixed = trade['sl_fixed']
    if direction == 'long'  and new_trail <= sl_fixed:
        trade['best_swing'] = best_swing
        return False
    if direction == 'short' and new_trail >= sl_fixed:
        trade['best_swing'] = best_swing
        return False

    # Trail mejoró y supera el SL fijo — actualizar STOP_MARKET
    old_trail = trade.get('trail_stop')
    if old_trail is not None:
        if direction == 'long'  and new_trail <= old_trail: return False
        if direction == 'short' and new_trail >= old_trail: return False

    log.info(f"Trail actualizado: {old_trail} → {new_trail:.4f} (swing={best_swing:.4f})")

    # Cancelar stop anterior y colocar nuevo
    cancel_stop_order(trade['symbol'], trade.get('stop_order_id'))
    new_stop_id = place_stop_order(trade['symbol'], direction, trade['qty'], new_trail)

    trade['best_swing']    = best_swing
    trade['trail_stop']    = new_trail
    trade['stop_order_id'] = new_stop_id
    trade['active_stop']   = new_trail
    return True

# ─── FORMATO MENSAJES TELEGRAM ────────────────────────────────────────────────
def fmt_open(trade: dict, balance: float) -> str:
    env  = '🧪 TESTNET' if USE_TESTNET else '🔴 REAL'
    dir_emoji = '🟢' if trade['direction'] == 'long' else '🔴'
    return (
        f"----------------------------\n"
        f"⚡ ENTRADA {env}\n"
        f"----------------------------\n"
        f"Par:      {trade['symbol']}\n"
        f"Dir:      {dir_emoji} {trade['direction'].upper()}\n"
        f"Precio:   {trade['entry']:,.4f}\n"
        f"Qty:      {trade['qty']}\n"
        f"SL fijo:  {trade['sl_fixed']:,.4f} (-{SL_PCT}%)\n"
        f"Trail:    {TRAIL_PCT}% del swing, actualiza por cierre 15m\n"
        f"Capital:  ${balance:,.2f} x {LEVERAGE}x\n"
        f"----------------------------"
    )

def fmt_close(trade: dict, exit_price: float, reason: str) -> str:
    entry    = trade['entry']
    direction = trade['direction']
    if direction == 'long':
        pnl_pct = (exit_price - entry) / entry * 100 * LEVERAGE
    else:
        pnl_pct = (entry - exit_price) / entry * 100 * LEVERAGE
    comm    = 0.05 * 2  # 0.05% por lado x2
    pnl_net = pnl_pct - comm * LEVERAGE
    balance = get_balance()
    pnl_usdt = balance * (CAPITAL_PCT/100) * pnl_net / 100

    result = '✅ WIN' if pnl_net > 0 else '❌ LOSS'
    dir_emoji = '🟢' if direction == 'long' else '🔴'
    reason_str = '📍 Trailing stop (Binance STOP_MARKET)' if reason == 'trailing' else '🛑 Stop loss (Binance STOP_MARKET)'

    dur = datetime.now(timezone.utc) - trade['opened_at']
    h, rem = divmod(int(dur.total_seconds()), 3600)
    m = rem // 60

    return (
        f"----------------------------\n"
        f"🔔 SALIDA - {result}\n"
        f"----------------------------\n"
        f"{trade['symbol']} {dir_emoji} {direction.upper()}\n"
        f"Motivo:   {reason_str}\n"
        f"Entrada:  {entry:,.4f}\n"
        f"Salida:   {exit_price:,.4f}\n"
        f"SL fijo:  {trade['sl_fixed']:,.4f}\n"
        f"Trail:    {trade['trail_stop']:,.4f}" + ("\n" if trade['trail_stop'] else " N/A\n") +
        f"P/L:      {pnl_net:+.3f}% ({pnl_usdt:+.2f} USDT)\n"
        f"Duracion: {h}h {m:02d}m\n"
        f"----------------------------"
    )

# ─── TELEGRAM HELPERS ─────────────────────────────────────────────────────────
async def send_tg(app: Application, text: str) -> None:
    try:
        await app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text, parse_mode='Markdown')
    except Exception as e:
        log.error(f"Telegram Markdown error: {e} — reintentando sin formato")
        try:
            plain = text.replace('*','').replace('`','')
            await app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=plain)
        except Exception as e2:
            log.error(f"Telegram fallback error: {e2}")

# ─── COMANDOS TELEGRAM ────────────────────────────────────────────────────────
async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    env = '🧪 TESTNET' if USE_TESTNET else '🔴 REAL'
    await update.message.reply_text(
        f"🤖 *Brújula Bot EMA15m* {env}\n\n"
        f"Señal: 15m — EMA{EMA_PERIOD} + ADX{ADX_PERIOD}>={ADX_MIN}\n"
        f"Entrada: open de la vela siguiente\n"
        f"Stop: SL {SL_PCT}% fijo + Trail {TRAIL_PCT}% swing 15m\n"
        f"Par: {SYMBOL} · Lev: {LEVERAGE}× · Capital: {CAPITAL_PCT}%\n"
        f"Scan: cada {SCAN_INTERVAL}s\n\n"
        f"/help para comandos",
        parse_mode='Markdown'
    )

async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "/status — posición activa\n"
        "/close  — cerrar posición manualmente\n"
        "/balance — balance USDT\n"
    )

async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    global active_trade
    if not active_trade:
        await update.message.reply_text("📭 Sin posición activa.")
        return
    t = active_trade
    try:
        mark = get_mark_price(t['symbol'])
        if t['direction'] == 'long':
            pnl = (mark - t['entry']) / t['entry'] * 100 * LEVERAGE
        else:
            pnl = (t['entry'] - mark) / t['entry'] * 100 * LEVERAGE
        active_stop = t.get('trail_stop') or t['sl_fixed']
        dur = datetime.now(timezone.utc) - t['opened_at']
        h, rem = divmod(int(dur.total_seconds()), 3600)
        m = rem // 60
        await update.message.reply_text(
            f"📊 Posición activa\n\n"
            f"{t['symbol']} {'🟢' if t['direction']=='long' else '🔴'} {t['direction'].upper()}\n"
            f"Entrada:      {t['entry']:,.4f}\n"
            f"Precio actual:{mark:,.4f}\n"
            f"P/L actual:   {pnl:+.3f}%\n"
            f"SL fijo:      {t['sl_fixed']:,.4f}\n"
            f"Mejor swing:  {t['best_swing']:,.4f}\n"
            f"Trail stop:   {t.get('trail_stop', 'pendiente')}\n"
            f"Stop activo:  {active_stop:,.4f}\n"
            f"Stop Binance: {t.get('stop_order_id', 'None')}\n"
            f"Duración:     {h}h {m:02d}m"
        )
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")

async def cmd_close(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    global active_trade
    if not active_trade:
        await update.message.reply_text("📭 Sin posición activa.")
        return
    exit_price = close_position_market(active_trade)
    if exit_price:
        msg = fmt_close(active_trade, exit_price, 'manual')
        active_trade = None
        await send_tg(ctx.application, msg)
    else:
        await update.message.reply_text("❌ Error cerrando posición.")

async def cmd_balance(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    bal = get_balance()
    await update.message.reply_text(f"💰 Balance disponible: ${bal:,.2f} USDT")

# ─── SCAN PRINCIPAL ───────────────────────────────────────────────────────────
async def scan(app: Application) -> None:
    global active_trade, last_15m_candle

    # ── 1. GESTIONAR TRADE ABIERTO ────────────────────────────────────────────
    if active_trade:

        # 1a. Stop software como RESPALDO (si Binance no tiene stop real)
        if not active_trade.get('stop_order_id'):
            try:
                mark = get_mark_price(active_trade['symbol'])
                active_stop = active_trade.get('trail_stop') or active_trade['sl_fixed']
                direction   = active_trade['direction']
                sl_tocado   = (
                    (direction == 'long'  and mark <= active_stop) or
                    (direction == 'short' and mark >= active_stop)
                )
                if sl_tocado:
                    log.info(f"Stop software (respaldo) tocado: mark={mark:.4f} vs stop={active_stop:.4f}")
                    reason     = 'trailing' if active_trade.get('trail_stop') else 'sl'
                    exit_price = close_position_market(active_trade)
                    if exit_price is None:
                        exit_price = mark
                    msg = fmt_close(active_trade, exit_price, reason)
                    active_trade = None
                    await send_tg(app, msg)
                    last_15m_candle = None
                    return
            except Exception as e:
                log.error(f"Error verificando stop software: {e}")

        # 1b. ¿Binance cerró la posición? (STOP_MARKET ejecutada)
        try:
            positions = get_client().futures_position_information(symbol=SYMBOL)
            pos_amt   = float(positions[0]['positionAmt']) if positions else 0.0
            if abs(pos_amt) < 0.001:
                log.info("Posición cerrada por Binance (STOP_MARKET ejecutada)")
                try:
                    mark = get_mark_price(active_trade['symbol'])
                except:
                    mark = active_trade['entry']
                reason     = 'trailing' if active_trade.get('trail_stop') else 'sl'
                exit_price = active_trade.get('trail_stop') or active_trade['sl_fixed']
                msg = fmt_close(active_trade, exit_price, reason)
                active_trade = None
                await send_tg(app, msg)
                last_15m_candle = None
                return
        except Exception as e:
            log.error(f"Error verificando posición: {e}")

        # 1c. Actualizar trailing con nuevas velas 15m
        try:
            df15 = get_klines(SYMBOL, '15m', limit=100)
            trail_updated = update_trail_15m(active_trade, df15)
            if trail_updated:
                log.info(f"Trail actualizado a {active_trade['trail_stop']:.4f}")
            else:
                log.info(
                    f"Trade {active_trade['direction'].upper()} | "
                    f"entry={active_trade['entry']:.4f} "
                    f"stop_activo={active_trade.get('trail_stop') or active_trade['sl_fixed']:.4f} "
                    f"trail={'pendiente' if not active_trade.get('trail_stop') else active_trade['trail_stop']:.4f if active_trade.get('trail_stop') else ''} "
                    f"stop_id={active_trade.get('stop_order_id','None')}"
                )
        except Exception as e:
            log.error(f"Error actualizando trailing: {e}")
        return

    # ── 2. SIN TRADE — BUSCAR SEÑAL ───────────────────────────────────────────
    try:
        df15 = get_klines(SYMBOL, '15m', limit=EMA_PERIOD + ADX_PERIOD * 3 + 10)

        # Anti-reentrada: una evaluación por vela 15m
        current_15m = int(df15['open_time'].iloc[-2])
        if current_15m == last_15m_candle:
            log.info("Vela 15m ya evaluada")
            return
        last_15m_candle = current_15m

        # Verificar señal
        signal, entry_price = check_signal_15m(df15)

        if not signal:
            log.info("Sin señal 15m")
            return

        log.info(f"Señal 15m: {signal.upper()} — EMA{EMA_PERIOD} + ADX{ADX_PERIOD}>={ADX_MIN}")

        # Abrir posición al open de la vela siguiente
        # La vela siguiente ya está en curso — entramos al open actual
        trade = open_position(SYMBOL, signal)
        if not trade:
            log.error("No se pudo abrir la posición")
            return

        active_trade = trade
        balance = get_balance()
        msg = fmt_open(trade, balance)
        await send_tg(app, msg)
        log.info(
            f"ABIERTO: {signal.upper()} {trade['qty']} {SYMBOL} @ {trade['entry']:.4f} "
            f"SL={trade['sl_fixed']:.4f} stop_id={trade.get('stop_order_id')}"
        )

    except Exception as e:
        log.error(f"Error en scan: {e}")

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main() -> None:
    env = '🧪 TESTNET' if USE_TESTNET else '🔴 REAL'
    log.info(f"Brújula Bot EMA15m arrancando — {env}")

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler('start',   cmd_start))
    app.add_handler(CommandHandler('help',    cmd_help))
    app.add_handler(CommandHandler('status',  cmd_status))
    app.add_handler(CommandHandler('close',   cmd_close))
    app.add_handler(CommandHandler('balance', cmd_balance))

    import asyncio
    from apscheduler.schedulers.asyncio import AsyncIOScheduler

    async def run():
        await app.initialize()
        await app.start()
        await app.bot.set_my_commands([
            ('start',   'Estado del bot'),
            ('status',  'Posición activa'),
            ('close',   'Cerrar posición'),
            ('balance', 'Balance USDT'),
            ('help',    'Ayuda'),
        ])

        # Mensaje de inicio
        try:
            env = '🧪 TESTNET' if USE_TESTNET else '🔴 REAL'
            await app.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=(
                    f"🤖 *Brújula Bot EMA15m* {env}\n\n"
                    f"Señal: 15m — EMA{EMA_PERIOD} + ADX{ADX_PERIOD} >= {ADX_MIN}\n"
                    f"Entrada: open de la vela siguiente\n"
                    f"Stop: SL {SL_PCT}% fijo + Trail {TRAIL_PCT}% swing 15m\n"
                    f"Par: {SYMBOL} | Lev: {LEVERAGE}x | Capital: {CAPITAL_PCT}%\n"
                    f"Scan: cada {SCAN_INTERVAL}s"
                ),
                parse_mode='Markdown'
            )
        except Exception as e:
            log.error(f"Error mensaje inicio: {e}")
            try:
                await app.bot.send_message(
                    chat_id=TELEGRAM_CHAT_ID,
                    text=f"Brújula Bot EMA15m arrancando — {'TESTNET' if USE_TESTNET else 'REAL'}"
                )
            except: pass

        loop = asyncio.get_event_loop()
        scheduler = AsyncIOScheduler()
        scheduler.add_job(
            lambda: asyncio.run_coroutine_threadsafe(scan(app), loop),
            'interval', seconds=SCAN_INTERVAL,
            id='scan', replace_existing=True
        )
        scheduler.start()
        log.info(f"Scheduler activo — scan cada {SCAN_INTERVAL}s")

        await app.updater.start_polling(drop_pending_updates=True)
        await asyncio.Event().wait()

    asyncio.run(run())

if __name__ == '__main__':
    main()
