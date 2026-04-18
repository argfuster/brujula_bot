"""
Brújula Trading Bot — Nivel 2
Semi-automático: detecta señales MTF, alerta por Telegram, espera confirmación, ejecuta en Binance Futures.
"""

import os
import asyncio
import logging
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import requests

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes
from binance.client import Client
from binance.enums import *

# ─── CONFIG ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
log = logging.getLogger(__name__)

TELEGRAM_TOKEN   = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')
BINANCE_API_KEY  = os.environ.get('BINANCE_API_KEY', '')
BINANCE_SECRET   = os.environ.get('BINANCE_API_SECRET', '')

SYMBOL      = os.environ.get('TRADING_SYMBOL', 'BTCUSDT')
LEVERAGE    = int(os.environ.get('LEVERAGE', '5'))
RISK_USDT   = float(os.environ.get('RISK_USDT', '10'))   # USD en riesgo por operación
EMA_PERIOD  = int(os.environ.get('EMA_PERIOD', '200'))
VOL_MULT    = float(os.environ.get('VOL_MULT', '1.5'))
RR_TARGET   = float(os.environ.get('RR_TARGET', '1.5'))  # activa trailing
TRAIL_R     = float(os.environ.get('TRAIL_R', '0.5'))    # distancia trailing en R
CONFIRM_MIN = int(os.environ.get('CONFIRM_MINUTES', '60'))

# ─── STATE ───────────────────────────────────────────────────────────────────
pending_signals: dict = {}    # signal_id → signal dict
active_trades:   dict = {}    # symbol → trade dict

# ─── BINANCE DATA ─────────────────────────────────────────────────────────────
def fetch_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    url = 'https://api.binance.com/api/v3/klines'
    r = requests.get(url, params={'symbol': symbol, 'interval': interval, 'limit': limit}, timeout=10)
    r.raise_for_status()
    df = pd.DataFrame(r.json(), columns=[
        'time','open','high','low','close','vol',
        'close_time','qv','trades','taker_buy_base','taker_buy_quote','ignore'
    ])
    for col in ['open','high','low','close','vol']:
        df[col] = df[col].astype(float)
    df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True)
    return df

def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift()).abs()
    lc = (df['low']  - df['close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

# ─── SETUP DETECTORS ─────────────────────────────────────────────────────────
def detect_stop_hunt(df15: pd.DataFrame, ema15: pd.Series, atr15: pd.Series, i: int) -> dict | None:
    if i < 12: return None
    cur  = df15.iloc[i]
    emaV = ema15.iloc[i]
    atrV = atr15.iloc[i]
    if pd.isna(emaV) or pd.isna(atrV): return None

    win     = df15.iloc[i-10:i]
    r_low   = win['low'].min()
    r_high  = win['high'].max()
    r_size  = r_high - r_low
    if r_size < atrV * 0.4: return None

    bull = cur['low'] < r_low and cur['close'] > r_low * 1.001 and cur['close'] > emaV * 0.998
    bear = cur['high'] > r_high and cur['close'] < r_high * 0.999 and cur['close'] < emaV * 1.002

    if bull and (r_low - cur['low']) > atrV * 0.25:
        return {'type': 'SH', 'dir': 'long',  'level': r_low,  'atr': atrV}
    if bear and (cur['high'] - r_high) > atrV * 0.25:
        return {'type': 'SH', 'dir': 'short', 'level': r_high, 'atr': atrV}
    return None

def detect_brt(df15: pd.DataFrame, ema15: pd.Series, i: int) -> dict | None:
    if i < 20: return None
    cur  = df15.iloc[i]
    emaV = ema15.iloc[i]
    if pd.isna(emaV): return None

    win    = df15.iloc[i-20:i]
    r_high = win['high'].max()
    r_low  = win['low'].min()

    if cur['close'] > emaV:
        near = abs(cur['close'] - r_high) / r_high < 0.004
        bull = cur['close'] > cur['open']
        if near and bull:
            return {'type': 'BRT', 'dir': 'long', 'level': r_high, 'atr': None}
    else:
        near = abs(cur['close'] - r_low) / r_low < 0.004
        bear = cur['close'] < cur['open']
        if near and bear:
            return {'type': 'BRT', 'dir': 'short', 'level': r_low, 'atr': None}
    return None

def detect_squeeze(df15: pd.DataFrame, ema15: pd.Series, atr15: pd.Series, i: int) -> dict | None:
    min_bars = 6
    if i < min_bars + 3: return None
    cur  = df15.iloc[i]
    emaV = ema15.iloc[i]
    atrV = atr15.iloc[i]
    if pd.isna(emaV) or pd.isna(atrV): return None

    win      = df15.iloc[i-min_bars:i]
    bodies   = (win['close'] - win['open']).abs()
    avg_body = bodies.mean()
    avg_vol  = win['vol'].mean()
    half     = min_bars // 2
    compress = bodies.iloc[half:].mean() < bodies.iloc[:half].mean() * 0.85
    bk_body  = abs(cur['close'] - cur['open'])

    if not compress: return None
    if bk_body < avg_body * 1.4: return None
    if cur['vol'] < avg_vol * VOL_MULT: return None

    if cur['close'] > cur['open'] and cur['close'] > emaV:
        return {'type': 'SQ', 'dir': 'long',  'atr': atrV}
    if cur['close'] < cur['open'] and cur['close'] < emaV:
        return {'type': 'SQ', 'dir': 'short', 'atr': atrV}
    return None

def check_5m_momentum(df5: pd.DataFrame, target_time, direction: str) -> bool:
    mask = df5['time'] >= target_time
    if not mask.any(): return False
    idx  = df5[mask].index[0]
    loc  = df5.index.get_loc(idx)
    if loc < 3: return False
    win  = df5.iloc[loc:min(loc+3, len(df5))]
    if len(win) == 0: return False
    mom  = win.iloc[-1]['close'] - win.iloc[0]['open']
    return mom > 0 if direction == 'long' else mom < 0

def is_weekend(dt) -> bool:
    return dt.weekday() >= 5  # 5=Sat, 6=Sun

# ─── MAIN SCAN ───────────────────────────────────────────────────────────────
def scan_for_signals(symbol: str) -> list[dict]:
    signals = []
    try:
        df1h  = fetch_klines(symbol, '1h',  EMA_PERIOD + 20)
        df15  = fetch_klines(symbol, '15m', EMA_PERIOD + 50)
        df5   = fetch_klines(symbol, '5m',  100)

        ema1h = calc_ema(df1h['close'], EMA_PERIOD)
        ema15 = calc_ema(df15['close'], EMA_PERIOD)
        atr15 = calc_atr(df15)

        # Weekend filter
        last_time = df15.iloc[-1]['time'].to_pydatetime()
        if is_weekend(last_time):
            log.info('Weekend — skipping scan')
            return []

        # Check last completed candle (i = -2, not -1 which is still forming)
        i = len(df15) - 2
        if i < EMA_PERIOD: return []

        candle_time = df15.iloc[i]['time']

        # Layer 1: 1h EMA bias
        nearest_1h = df1h.iloc[(df1h['time'] - candle_time).abs().argsort().iloc[0]]
        ema1h_val  = ema1h.iloc[df1h.index.get_loc(nearest_1h.name)]
        if pd.isna(ema1h_val): return []
        price = df15.iloc[i]['close']
        gap   = abs(price - ema1h_val) / ema1h_val
        if gap < 0.002: return []  # no clear bias (chop zone)

        # Layer 2: setup detection
        sig = None
        sig = sig or detect_stop_hunt(df15, ema15, atr15, i)
        sig = sig or detect_brt(df15, ema15, i)
        sig = sig or detect_squeeze(df15, ema15, atr15, i)
        if not sig: return []

        # Direction must align with 1h bias
        if sig['dir'] == 'long'  and price < ema1h_val * 0.997: return []
        if sig['dir'] == 'short' and price > ema1h_val * 1.003: return []

        # Layer 3: 5m momentum
        if not check_5m_momentum(df5, candle_time, sig['dir']): return []

        # Build signal
        atr_val = sig.get('atr') or atr15.iloc[i]
        risk    = atr_val * 1.5
        entry   = price
        stop    = entry - risk if sig['dir'] == 'long' else entry + risk
        target  = entry + risk * RR_TARGET if sig['dir'] == 'long' else entry - risk * RR_TARGET

        signal = {
            'id':        f"{symbol}_{sig['type']}_{int(candle_time.timestamp())}",
            'symbol':    symbol,
            'type':      sig['type'],
            'dir':       sig['dir'],
            'entry':     round(entry, 2),
            'stop':      round(stop, 2),
            'target':    round(target, 2),
            'risk':      round(risk, 2),
            'ema1h':     round(ema1h_val, 2),
            'candle_t':  candle_time,
            'detected':  datetime.now(timezone.utc),
        }
        signals.append(signal)
        log.info(f"Signal detected: {signal['type']} {signal['dir']} @ {signal['entry']}")

    except Exception as e:
        log.error(f"Scan error: {e}")

    return signals

# ─── TELEGRAM MESSAGES ───────────────────────────────────────────────────────
SETUP_EMOJI = {'SH': '🎯', 'BRT': '📐', 'SQ': '💥'}
DIR_EMOJI   = {'long': '🟢 LONG', 'short': '🔴 SHORT'}
TYPE_LABEL  = {'SH': 'Stop Hunt', 'BRT': 'Break & Retest', 'SQ': 'Squeeze'}

def format_signal_message(sig: dict) -> str:
    emoji   = SETUP_EMOJI.get(sig['type'], '🔔')
    pct_stop = abs(sig['entry'] - sig['stop']) / sig['entry'] * 100
    pct_tgt  = abs(sig['entry'] - sig['target']) / sig['entry'] * 100
    bias_dir = 'sobre' if sig['dir'] == 'long' else 'bajo'

    return (
        f"{emoji} *SEÑAL DETECTADA — {sig['symbol']}*\n\n"
        f"Setup:      `{TYPE_LABEL[sig['type']]}`\n"
        f"Dirección:  {DIR_EMOJI[sig['dir']]}\n"
        f"Entrada:    `{sig['entry']:,.2f}`\n"
        f"Stop:       `{sig['stop']:,.2f}`  _(−{pct_stop:.2f}%)_\n"
        f"Target 1.5R: `{sig['target']:,.2f}`  _(+{pct_tgt:.2f}%)_\n"
        f"Trailing:   `0.5R desde {sig['target']:,.2f}`\n\n"
        f"Sesgo 1h:   ✓ precio {bias_dir} EMA 200 ({sig['ema1h']:,.2f})\n\n"
        f"_Revisá orderflow en Exocharts antes de confirmar._\n"
        f"_Delta y CVD en tu dirección? Absorción en el nivel?_\n\n"
        f"⏱ Expira en {CONFIRM_MIN} minutos"
    )

def format_trade_update(sig: dict, event: str, price: float = 0, rr: float = 0) -> str:
    if event == 'opened':
        return (
            f"✅ *ORDEN EJECUTADA — {sig['symbol']}*\n\n"
            f"Setup: `{TYPE_LABEL[sig['type']]}` {DIR_EMOJI[sig['dir']]}\n"
            f"Entrada: `{price:,.2f}`\n"
            f"Stop: `{sig['stop']:,.2f}`\n"
            f"Trailing activo desde `{sig['target']:,.2f}` (1.5R)"
        )
    elif event == 'closed':
        emoji = '🏆' if rr > 0 else '🛑'
        return (
            f"{emoji} *POSICIÓN CERRADA — {sig['symbol']}*\n\n"
            f"Setup: `{TYPE_LABEL[sig['type']]}` {DIR_EMOJI[sig['dir']]}\n"
            f"Resultado: `{'+' if rr > 0 else ''}{rr:.2f}R`\n"
            f"Precio cierre: `{price:,.2f}`"
        )
    elif event == 'expired':
        return f"⏱ *Señal expirada* — {sig['symbol']} {sig['type']} {sig['dir']}"
    elif event == 'rejected':
        return f"❌ *Señal descartada* — {sig['symbol']} {sig['type']} {sig['dir']}"
    return ''

# ─── BINANCE EXECUTION ───────────────────────────────────────────────────────
TESTNET = os.environ.get('TESTNET', 'true').lower() == 'true'

def get_binance_client() -> Client:
    client = Client(BINANCE_API_KEY, BINANCE_SECRET, testnet=TESTNET)
    if TESTNET:
        client.FUTURES_URL = 'https://testnet.binancefuture.com/fapi'
    return client

def calc_qty(symbol: str, entry: float, risk_usdt: float, stop: float) -> float:
    """
    Calcula el tamaño de posición basado en el riesgo.
    qty = RISK_USDT / risk_per_unit
    El leverage no afecta el qty — solo el margen requerido.
    """
    risk_per_unit = abs(entry - stop)
    if risk_per_unit == 0:
        return 0
    qty = risk_usdt / risk_per_unit

    # Round to symbol precision
    client = get_binance_client()
    info   = client.futures_exchange_info()
    for s in info['symbols']:
        if s['symbol'] == symbol:
            for f in s['filters']:
                if f['filterType'] == 'LOT_SIZE':
                    step = float(f['stepSize'])
                    qty  = round(qty - (qty % step), 8)
                    return qty
    return round(qty, 3)

def execute_entry(sig: dict) -> dict | None:
    try:
        client = get_binance_client()
        client.futures_change_leverage(symbol=sig['symbol'], leverage=LEVERAGE)

        qty  = calc_qty(sig['symbol'], sig['entry'], RISK_USDT, sig['stop'])
        if qty <= 0:
            log.error(f"Invalid qty calculated: {qty}")
            return None

        # Verify margin available
        nocional = qty * sig['entry']
        margen   = nocional / LEVERAGE
        account  = client.futures_account()
        available = float(account.get('availableBalance', 0))
        if available < margen:
            log.error(f"Insufficient margin: need ${margen:.2f}, have ${available:.2f}")
            return None

        log.info(f"Order size: {qty} BTC | Nocional: ${nocional:.2f} | Margen: ${margen:.2f} | Disponible: ${available:.2f}")

        side = SIDE_BUY if sig['dir'] == 'long' else SIDE_SELL

        order = client.futures_create_order(
            symbol    = sig['symbol'],
            side      = side,
            type      = ORDER_TYPE_MARKET,
            quantity  = qty,
        )

        # Place stop loss
        sl_side = SIDE_SELL if sig['dir'] == 'long' else SIDE_BUY
        client.futures_create_order(
            symbol        = sig['symbol'],
            side          = sl_side,
            type          = 'STOP_MARKET',
            stopPrice     = sig['stop'],
            quantity      = qty,
            reduceOnly    = True,
        )

        fill_price = float(order.get('avgPrice', sig['entry']) or sig['entry'])
        log.info(f"Order executed: {sig['dir']} {qty} {sig['symbol']} @ {fill_price}")
        return {'order': order, 'qty': qty, 'fill_price': fill_price}

    except Exception as e:
        log.error(f"Execution error: {e}")
        return None

def close_position(symbol: str, qty: float, direction: str):
    try:
        client = get_binance_client()
        side   = SIDE_SELL if direction == 'long' else SIDE_BUY
        client.futures_create_order(
            symbol     = symbol,
            side       = side,
            type       = ORDER_TYPE_MARKET,
            quantity   = qty,
            reduceOnly = True,
        )
        log.info(f"Position closed: {symbol}")
    except Exception as e:
        log.error(f"Close error: {e}")

# ─── TRAILING MANAGER ─────────────────────────────────────────────────────────
async def manage_trailing(app, sig: dict, trade: dict):
    """
    Monitors position and manages trailing stop once price hits 1.5R target.
    Runs as async task until position closes.
    """
    symbol    = sig['symbol']
    direction = sig['dir']
    entry     = sig['entry']
    risk      = sig['risk']
    qty       = trade['qty']
    target    = sig['target']   # 1.5R activation price
    trail_gap = risk * TRAIL_R  # 0.5R in price terms

    phase      = 0   # 0=watching, 1=trailing active
    max_fav    = entry
    trail_stop = None
    client     = get_binance_client()

    log.info(f"Trailing manager started for {symbol} {direction}")

    while symbol in active_trades:
        await asyncio.sleep(30)  # check every 30 seconds
        try:
            ticker = client.futures_symbol_ticker(symbol=symbol)
            price  = float(ticker['price'])

            # Update max favorable
            if direction == 'long'  and price > max_fav: max_fav = price
            if direction == 'short' and price < max_fav: max_fav = price

            # Phase 0 → 1: hit target, activate trailing
            if phase == 0:
                hit = (direction == 'long' and price >= target) or \
                      (direction == 'short' and price <= target)
                if hit:
                    phase      = 1
                    trail_stop = max_fav - trail_gap if direction == 'long' else max_fav + trail_gap
                    log.info(f"Trailing activated @ {price:.2f}, trail_stop={trail_stop:.2f}")
                    # Cancel original SL, place new one at trail_stop
                    _update_stop(client, sig, trail_stop, qty)

            # Phase 1: update trailing
            elif phase == 1:
                new_stop = max_fav - trail_gap if direction == 'long' else max_fav + trail_gap
                if (direction == 'long'  and new_stop > trail_stop) or \
                   (direction == 'short' and new_stop < trail_stop):
                    trail_stop = new_stop
                    _update_stop(client, sig, trail_stop, qty)
                    log.info(f"Trail moved to {trail_stop:.2f} (max={max_fav:.2f})")

                # Check if stop was hit
                hit_stop = (direction == 'long'  and price <= trail_stop) or \
                           (direction == 'short' and price >= trail_stop)
                if hit_stop:
                    rr = (trail_stop - entry) / risk if direction == 'long' \
                         else (entry - trail_stop) / risk
                    active_trades.pop(symbol, None)
                    await app.bot.send_message(
                        chat_id = TELEGRAM_CHAT_ID,
                        text    = format_trade_update(sig, 'closed', trail_stop, round(rr, 2)),
                        parse_mode = 'Markdown'
                    )
                    log.info(f"Position closed by trailing @ {trail_stop:.2f} ({rr:.2f}R)")
                    return

            # Check if original SL hit (phase 0)
            elif phase == 0:
                hit_sl = (direction == 'long'  and price <= sig['stop']) or \
                         (direction == 'short' and price >= sig['stop'])
                if hit_sl:
                    rr = -1.0
                    active_trades.pop(symbol, None)
                    await app.bot.send_message(
                        chat_id = TELEGRAM_CHAT_ID,
                        text    = format_trade_update(sig, 'closed', sig['stop'], rr),
                        parse_mode = 'Markdown'
                    )
                    return

        except Exception as e:
            log.error(f"Trailing error: {e}")

def _update_stop(client, sig: dict, new_stop: float, qty: float):
    """Cancel all open SL orders and place new one."""
    try:
        orders = client.futures_get_open_orders(symbol=sig['symbol'])
        for o in orders:
            if o.get('type') in ('STOP_MARKET', 'STOP') and o.get('reduceOnly'):
                client.futures_cancel_order(symbol=sig['symbol'], orderId=o['orderId'])
        sl_side = SIDE_SELL if sig['dir'] == 'long' else SIDE_BUY
        client.futures_create_order(
            symbol     = sig['symbol'],
            side       = sl_side,
            type       = 'STOP_MARKET',
            stopPrice  = round(new_stop, 2),
            quantity   = qty,
            reduceOnly = True,
        )
    except Exception as e:
        log.error(f"Stop update error: {e}")

# ─── TELEGRAM HANDLERS ────────────────────────────────────────────────────────
async def handle_confirmation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query   = update.callback_query
    await query.answer()
    data    = query.data  # "confirm_SIGNAL_ID" or "reject_SIGNAL_ID"
    parts   = data.split('_', 1)
    action  = parts[0]
    sig_id  = parts[1] if len(parts) > 1 else ''
    sig     = pending_signals.pop(sig_id, None)

    if not sig:
        await query.edit_message_text("⚠️ Señal expirada o ya procesada.")
        return

    if action == 'reject':
        await query.edit_message_text(format_trade_update(sig, 'rejected'), parse_mode='Markdown')
        return

    # Execute
    await query.edit_message_text(f"⏳ Ejecutando orden {sig['dir'].upper()} en {sig['symbol']}...")
    trade = execute_entry(sig)

    if not trade:
        await context.bot.send_message(
            chat_id    = TELEGRAM_CHAT_ID,
            text       = f"❌ Error ejecutando orden. Verificá manualmente en Binance.",
            parse_mode = 'Markdown'
        )
        return

    active_trades[sig['symbol']] = {**sig, **trade}
    await context.bot.send_message(
        chat_id    = TELEGRAM_CHAT_ID,
        text       = format_trade_update(sig, 'opened', trade['fill_price']),
        parse_mode = 'Markdown'
    )

    # Start trailing manager
    asyncio.create_task(manage_trailing(context.application, sig, trade))

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not active_trades:
        await update.message.reply_text("📭 Sin posiciones abiertas.")
        return
    msg = "📊 *Posiciones abiertas:*\n\n"
    client = get_binance_client()
    for sym, t in active_trades.items():
        try:
            price = float(client.futures_symbol_ticker(symbol=sym)['price'])
            rr    = (price - t['entry']) / t['risk'] if t['dir'] == 'long' \
                    else (t['entry'] - price) / t['risk']
            msg  += f"• {sym} {DIR_EMOJI[t['dir']]} | Precio: `{price:,.2f}` | R actual: `{rr:+.2f}R`\n"
        except:
            msg += f"• {sym} {DIR_EMOJI[t['dir']]}\n"
    await update.message.reply_text(msg, parse_mode='Markdown')

async def cmd_close(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not active_trades:
        await update.message.reply_text("Sin posiciones abiertas.")
        return
    for sym, t in list(active_trades.items()):
        close_position(sym, t['qty'], t['dir'])
        active_trades.pop(sym, None)
        await update.message.reply_text(f"✅ Posición {sym} cerrada manualmente.")

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 *Brújula Bot — Comandos*\n\n"
        "/status — ver posiciones abiertas\n"
        "/close  — cerrar todas las posiciones\n"
        "/help   — este mensaje\n\n"
        "_Las señales llegan automáticamente cada 15 min._",
        parse_mode='Markdown'
    )

# ─── SCAN JOB ────────────────────────────────────────────────────────────────
async def scan_job(app):
    log.info("Running scan...")
    signals = scan_for_signals(SYMBOL)

    for sig in signals:
        # Skip if signal for this symbol already pending or in trade
        if sig['id'] in pending_signals: continue
        if sig['symbol'] in active_trades: continue

        pending_signals[sig['id']] = sig

        keyboard = InlineKeyboardMarkup([[
            InlineKeyboardButton("✅ Entrar", callback_data=f"confirm_{sig['id']}"),
            InlineKeyboardButton("❌ Descartar", callback_data=f"reject_{sig['id']}"),
        ]])

        try:
            await app.bot.send_message(
                chat_id      = TELEGRAM_CHAT_ID,
                text         = format_signal_message(sig),
                parse_mode   = 'Markdown',
                reply_markup = keyboard,
            )
        except Exception as e:
            log.error(f"Telegram send error: {e}")

    # Expire old pending signals
    now = datetime.now(timezone.utc)
    expired = [
        k for k, v in pending_signals.items()
        if (now - v['detected']).total_seconds() > CONFIRM_MIN * 60
    ]
    for k in expired:
        sig = pending_signals.pop(k)
        try:
            await app.bot.send_message(
                chat_id    = TELEGRAM_CHAT_ID,
                text       = format_trade_update(sig, 'expired'),
                parse_mode = 'Markdown'
            )
        except:
            pass

# ─── MAIN ────────────────────────────────────────────────────────────────────
async def scan_callback(context):
    """JobQueue callback — runs in the correct event loop."""
    await scan_job(context.application)

async def post_init(app):
    app.job_queue.run_repeating(
        callback = scan_callback,
        interval = 900,
        first    = 15,
        name     = 'scan',
    )
    log.info(f"Bot started — monitoring {SYMBOL} | Risk: ${RISK_USDT} | Leverage: {LEVERAGE}x")

def main():
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        raise ValueError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are required")

    app = (Application.builder()
           .token(TELEGRAM_TOKEN)
           .post_init(post_init)
           .build())

    app.add_handler(CommandHandler('status', cmd_status))
    app.add_handler(CommandHandler('close',  cmd_close))
    app.add_handler(CommandHandler('help',   cmd_help))
    app.add_handler(CallbackQueryHandler(handle_confirmation))
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

if __name__ == '__main__':
    main()
