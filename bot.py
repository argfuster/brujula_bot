"""
Brújula Trading Bot — EMA + ADX + ATR + Trailing Swing
=======================================================
Variables Railway:
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, BINANCE_API_KEY, BINANCE_API_SECRET
  USE_TESTNET (true), TRADING_SYMBOL (ETHUSDT), LEVERAGE (2), CAPITAL_PCT (95)
  EMA_PERIOD (25), ADX_PERIOD (14), ADX_MIN (25)
  ATR_PERIOD (14), ATR_MIN_PCT (0.25)
  SL_PCT (0.5), TRAIL_PCT (50), SCAN_INTERVAL (60)

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

# ─── LOGGING ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
log = logging.getLogger(__name__)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')
BINANCE_KEY      = os.environ.get('BINANCE_API_KEY', '')
BINANCE_SECRET   = os.environ.get('BINANCE_API_SECRET', '')
USE_TESTNET      = os.environ.get('USE_TESTNET', 'true').lower() == 'true'
SYMBOL           = os.environ.get('TRADING_SYMBOL', 'ETHUSDT')
LEVERAGE         = int(os.environ.get('LEVERAGE', '2'))
CAPITAL_PCT      = float(os.environ.get('CAPITAL_PCT', '95'))
EMA_PERIOD       = int(os.environ.get('EMA_PERIOD', '25'))
ADX_PERIOD       = int(os.environ.get('ADX_PERIOD', '14'))
ADX_MIN          = float(os.environ.get('ADX_MIN', '25'))
ATR_PERIOD       = int(os.environ.get('ATR_PERIOD', '14'))
ATR_MIN_PCT      = float(os.environ.get('ATR_MIN_PCT', '0.25'))
SL_PCT           = float(os.environ.get('SL_PCT', '0.5'))
TRAIL_PCT        = float(os.environ.get('TRAIL_PCT', '50'))
SCAN_INTERVAL    = int(os.environ.get('SCAN_INTERVAL', '60'))
TF               = '15m'

# ─── ESTADO GLOBAL ────────────────────────────────────────────────────────────
active_trade: dict | None = None
pending_signal: str | None = None
last_candle_time: int | None = None

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

def get_klines(symbol: str, limit: int = 350) -> pd.DataFrame:
    raw = get_client().futures_klines(symbol=symbol, interval=TF, limit=limit)
    df = pd.DataFrame(raw, columns=[
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
    up   = df['high'].diff()
    down = -df['low'].diff()
    pdm  = np.where((up > down) & (up > 0), up, 0.0)
    ndm  = np.where((down > up) & (down > 0), down, 0.0)
    atr  = calc_atr(df, period)
    pdi  = 100 * pd.Series(pdm, index=df.index).ewm(span=period, adjust=False).mean() / atr
    ndi  = 100 * pd.Series(ndm, index=df.index).ewm(span=period, adjust=False).mean() / atr
    dx   = (100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan))
    return dx.ewm(span=period, adjust=False).mean()

# ─── HORARIO ──────────────────────────────────────────────────────────────────
def in_session(ts_sec: int) -> bool:
    d   = datetime.fromtimestamp(ts_sec, tz=timezone.utc)
    hm  = d.hour * 60 + d.minute
    dow = d.weekday()
    if dow >= 5:                                    return True   # finde
    if 13 * 60 + 30 <= hm < 20 * 60:               return True   # NY
    if 8  * 60       <= hm < 12 * 60:               return True   # Londres
    if 12 * 60       <= hm < 13 * 60 + 30:          return True   # Pre-NY
    if hm < 8 * 60:                                 return True   # Asia
    return False                                                  # Post-NY: OFF

# ─── SEÑAL ────────────────────────────────────────────────────────────────────
def check_signal(df: pd.DataFrame) -> str | None:
    min_bars = max(EMA_PERIOD, ADX_PERIOD * 3, ATR_PERIOD) + 10
    if len(df) < min_bars:
        return None

    ema = calc_ema(df['close'], EMA_PERIOD)
    atr = calc_atr(df, ATR_PERIOD)
    adx = calc_adx(df, ADX_PERIOD)

    i     = -2
    close = df['close'].iloc[i]
    ema_v = ema.iloc[i]
    atr_v = atr.iloc[i]
    adx_v = adx.iloc[i]

    if any(pd.isna(x) for x in [ema_v, atr_v, adx_v]) or close <= 0:
        return None
    if adx_v < ADX_MIN:
        return None
    if atr_v / close * 100 < ATR_MIN_PCT:
        return None

    if close > ema_v:   return 'long'
    if close < ema_v:   return 'short'
    return None

# ─── GESTIÓN DE POSICIÓN ──────────────────────────────────────────────────────
def check_exit(df: pd.DataFrame, trade: dict) -> tuple[bool, str, float]:
    ema   = calc_ema(df['close'], EMA_PERIOD)
    i     = -2
    close = df['close'].iloc[i]
    ema_v = ema.iloc[i]
    prev_close = df['close'].iloc[i - 1]

    direction  = trade['direction']
    entry      = trade['entry']
    sl_fixed   = trade['sl_fixed']

    # Actualizar mejor cierre histórico (trailing real — nunca retrocede)
    if direction == 'long' and prev_close > trade['best_swing']:
        trade['best_swing'] = prev_close
    elif direction == 'short' and prev_close < trade['best_swing']:
        trade['best_swing'] = prev_close

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

    # Exit por stop (close de vela)
    if direction == 'long' and close <= active_stop:
        reason = 'trailing' if (trail_stop and trail_stop >= sl_fixed) else 'sl'
        return True, reason, active_stop
    if direction == 'short' and close >= active_stop:
        reason = 'trailing' if (trail_stop and trail_stop <= sl_fixed) else 'sl'
        return True, reason, active_stop

    # Exit por cruce EMA
    if direction == 'long'  and close < ema_v:  return True, 'ema', close
    if direction == 'short' and close > ema_v:  return True, 'ema', close

    return False, '', 0.0

# ─── EJECUCIÓN ────────────────────────────────────────────────────────────────
def open_position(direction: str) -> dict | None:
    try:
        client = get_client()
        client.futures_change_leverage(symbol=SYMBOL, leverage=LEVERAGE)
        balance = get_balance()
        price   = get_mark_price(SYMBOL)
        step    = get_step_size(SYMBOL)

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
    h, m     = int(dur.total_seconds() // 3600), int((dur.total_seconds() % 3600) // 60)
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
    global active_trade, pending_signal
    if not active_trade:
        msg = "📭 *Sin posición abierta.*"
        if pending_signal:
            msg += f"\n⏳ Señal pendiente: {pending_signal.upper()} (entra en próxima vela)"
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
        f"🤖 *Brújula EMA+ADX+ATR Bot* {env}\n\n"
        f"/status — posición activa y stops\n"
        f"/close  — cerrar manualmente\n"
        f"/help   — este mensaje\n\n"
        f"EMA `{EMA_PERIOD}` · ADX `{ADX_MIN}` · ATR `{ATR_MIN_PCT}%` · `{TF}`\n"
        f"SL `{SL_PCT}%` + Trailing `{TRAIL_PCT}%` del swing\n"
        f"`{SYMBOL}` · `{LEVERAGE}×` · Capital `{CAPITAL_PCT}%`\n"
        f"Sesiones: NY · Londres · Pre-NY · Asia · Finde\n"
        f"Scan cada `{SCAN_INTERVAL}s`"
    )
    await update.message.reply_text(msg, parse_mode='Markdown')

# ─── SCAN ─────────────────────────────────────────────────────────────────────
async def scan_job(app: Application) -> None:
    global active_trade, pending_signal, last_candle_time

    try:
        df = get_klines(SYMBOL, limit=350)
    except Exception as e:
        log.error(f"Error klines: {e}")
        return

    # Deduplicar por vela
    current_candle = int(df['open_time'].iloc[-2])
    if current_candle == last_candle_time:
        log.info("Vela ya evaluada")
        return
    last_candle_time = current_candle

    # ── GESTIÓN DE POSICIÓN ABIERTA ──────────────────────────────────────────
    if active_trade:
        try:
            should_exit, reason, _ = check_exit(df, active_trade)
            if should_exit:
                exit_price = close_position(active_trade)
                if exit_price:
                    msg = fmt_close(active_trade, exit_price, reason)
                    active_trade   = None
                    pending_signal = None
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

    # ── EJECUTAR SEÑAL PENDIENTE (open de esta vela) ─────────────────────────
    if pending_signal:
        log.info(f"Ejecutando señal pendiente: {pending_signal.upper()}")
        trade = open_position(pending_signal)
        pending_signal = None
        if trade:
            active_trade = trade
            await send_tg(app, fmt_open(trade))
        else:
            log.error("Fallo en apertura")
        return

    # ── HORARIO ──────────────────────────────────────────────────────────────
    ts_vela = int(df['open_time'].iloc[-2]) // 1000  # Binance devuelve ms, convertir a segundos
    if not in_session(ts_vela):
        log.info(f"Fuera de horario ({datetime.fromtimestamp(ts_vela, tz=timezone.utc).strftime('%H:%M UTC')})")
        return

    # ── DETECTAR SEÑAL ───────────────────────────────────────────────────────
    try:
        signal = check_signal(df)
    except Exception as e:
        log.error(f"Error check_signal: {e}")
        return

    if signal:
        log.info(f"Señal: {signal.upper()} — entra próxima vela")
        pending_signal = signal
    else:
        log.info("Sin señal")

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
    log.info(f"Bot iniciado — {SYMBOL} {TF} | {env} | Lev:{LEVERAGE}x Cap:{CAPITAL_PCT}% EMA:{EMA_PERIOD} ADX:{ADX_MIN} ATR:{ATR_MIN_PCT}% SL:{SL_PCT}% Trail:{TRAIL_PCT}%")
    await app.bot.send_message(
        chat_id=TELEGRAM_CHAT_ID,
        parse_mode='Markdown',
        text=(
            f"🤖 *Brújula EMA+ADX+ATR Bot* {'🧪 TESTNET' if USE_TESTNET else '🔴 REAL'}\n\n"
            f"*Par:* `{SYMBOL}` · *TF:* `{TF}` · *Leverage:* `{LEVERAGE}×`\n"
            f"*EMA:* `{EMA_PERIOD}` · *ADX:* `{ADX_MIN}` · *ATR:* `{ATR_MIN_PCT}%`\n"
            f"*SL:* `{SL_PCT}%` fijo + trailing `{TRAIL_PCT}%` del swing\n"
            f"*Capital:* `{CAPITAL_PCT}%` por trade\n"
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
