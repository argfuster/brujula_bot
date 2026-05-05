"""
Brújula Trading Bot — EMA Stack
================================
Full automático: EMA15/60/200 + ADX + ATR + RSI + Volumen en 5m
Horario: NY (13:30-20:00 UTC) + Asia (00:00-08:00 UTC) + Fin de semana
Notifica por Telegram al abrir/cerrar — sin pedir confirmación
Sizing: usa balance completo × leverage (capitalización real)

Variables de entorno en Railway:
  TELEGRAM_BOT_TOKEN   (requerida)
  TELEGRAM_CHAT_ID     (requerida)
  BINANCE_API_KEY      (requerida)
  BINANCE_API_SECRET   (requerida)
  USE_TESTNET          → true / false   (default: true)
  TRADING_SYMBOL       → ETHUSDT        (default: ETHUSDT)
  LEVERAGE             → 2              (default: 2)
  EMA_FAST             → 15             (default: 15)
  EMA_MID              → 60             (default: 60)
  EMA_SLOW             → 200            (default: 200)
  ADX_MIN              → 30             (default: 30)
  ATR_MIN_PCT          → 0.2            (default: 0.2)
  RSI_LONG_MIN         → 55             (default: 55)
  RSI_SHORT_MAX        → 45             (default: 45)
  VOL_PERIOD           → 50             (default: 50)
  VOL_MULT             → 1.5            (default: 1.5)
  SCAN_INTERVAL        → 60             (default: 60)
"""

import os
import asyncio
import logging
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import requests

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from binance.client import Client
from binance.enums import SIDE_BUY, SIDE_SELL, ORDER_TYPE_MARKET

# ─── LOGGING ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
log = logging.getLogger(__name__)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')
BINANCE_API_KEY  = os.environ.get('BINANCE_API_KEY', '')
BINANCE_SECRET   = os.environ.get('BINANCE_API_SECRET', '')
USE_TESTNET      = os.environ.get('USE_TESTNET', 'true').lower() == 'true'

SYMBOL        = os.environ.get('TRADING_SYMBOL', 'ETHUSDT')
LEVERAGE      = int(os.environ.get('LEVERAGE', '2'))
EMA_FAST      = int(os.environ.get('EMA_FAST', '15'))
EMA_MID       = int(os.environ.get('EMA_MID', '60'))
EMA_SLOW      = int(os.environ.get('EMA_SLOW', '200'))
ADX_MIN       = float(os.environ.get('ADX_MIN', '30'))
ATR_MIN_PCT   = float(os.environ.get('ATR_MIN_PCT', '0.2'))
RSI_MIN       = float(os.environ.get('RSI_LONG_MIN', '55'))
RSI_MAX       = float(os.environ.get('RSI_SHORT_MAX', '45'))
VOL_PERIOD    = int(os.environ.get('VOL_PERIOD', '50'))
VOL_MULT      = float(os.environ.get('VOL_MULT', '1.5'))
SCAN_INTERVAL = int(os.environ.get('SCAN_INTERVAL', '60'))

ATR_PERIOD    = 14
ADX_PERIOD    = 14
RSI_PERIOD    = 14

# ─── ESTADO GLOBAL ────────────────────────────────────────────────────────────
active_trade: dict | None = None
last_candle_time: int | None = None  # open_time de la última vela evaluada

# ─── CLIENTE BINANCE ──────────────────────────────────────────────────────────
def get_client() -> Client:
    c = Client(BINANCE_API_KEY, BINANCE_SECRET, testnet=USE_TESTNET)
    if USE_TESTNET:
        c.FUTURES_URL = 'https://testnet.binancefuture.com/fapi'
    return c

def get_price(symbol: str) -> float:
    """Precio mark del símbolo. Retorna 0.0 si falla."""
    try:
        c = get_client()
        data = c.futures_mark_price(symbol=symbol)
        price = float(data.get('markPrice', 0))
        return price
    except Exception as e:
        log.error(f"Error obteniendo precio {symbol}: {e}")
        return 0.0

def get_klines(symbol: str, interval: str = '5m', limit: int = 300) -> pd.DataFrame:
    """Descarga velas y retorna DataFrame con columnas OHLCV."""
    c = get_client()
    raw = c.futures_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(raw, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'vol',
        'close_time', 'quote_vol', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    for col in ['open', 'high', 'low', 'close', 'vol']:
        df[col] = pd.to_numeric(df[col])
    return df

# ─── INDICADORES ──────────────────────────────────────────────────────────────
def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def calc_atr(df: pd.DataFrame, period: int) -> pd.Series:
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift(1)).abs()
    lc = (df['low']  - df['close'].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def calc_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_adx(df: pd.DataFrame, period: int) -> pd.Series:
    up   = df['high'].diff()
    down = -df['low'].diff()
    plus_dm  = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = calc_atr(df, period)
    tr_smooth   = pd.Series(plus_dm).ewm(span=period, adjust=False).mean()
    plus_di     = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean() / tr
    minus_di    = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean() / tr
    dx          = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx         = dx.ewm(span=period, adjust=False).mean()
    adx.index   = df.index
    return adx

# ─── SEÑAL ────────────────────────────────────────────────────────────────────
def check_signal(df: pd.DataFrame) -> str | None:
    """
    Retorna 'long', 'short' o None.
    Usa la vela cerrada anterior (iloc[-2]) para evitar señales de vela en formación.
    """
    limit_needed = max(EMA_SLOW, VOL_PERIOD, ADX_PERIOD * 2) + 10
    if len(df) < limit_needed:
        return None

    ema_fast = calc_ema(df['close'], EMA_FAST)
    ema_mid  = calc_ema(df['close'], EMA_MID)
    ema_slow = calc_ema(df['close'], EMA_SLOW)
    atr      = calc_atr(df, ATR_PERIOD)
    adx      = calc_adx(df, ADX_PERIOD)
    rsi      = calc_rsi(df['close'], RSI_PERIOD)

    i = -2  # última vela cerrada
    close   = df['close'].iloc[i]
    ef      = ema_fast.iloc[i]
    em      = ema_mid.iloc[i]
    es      = ema_slow.iloc[i]
    atr_val = atr.iloc[i]
    adx_val = adx.iloc[i]
    rsi_val = rsi.iloc[i]

    if any(pd.isna(x) for x in [ef, em, es, atr_val, adx_val, rsi_val]):
        return None

    # Filtro ADX
    if adx_val < ADX_MIN:
        return None

    # Filtro ATR%
    if close > 0 and (atr_val / close * 100) < ATR_MIN_PCT:
        return None

    # Filtro volumen
    avg_vol = df['vol'].iloc[i - VOL_PERIOD:i].mean()
    if avg_vol > 0 and df['vol'].iloc[i] < avg_vol * VOL_MULT:
        return None

    # Alineación de EMAs
    # Si EMA_SLOW=0 en Railway, ignora la EMA lenta — señal solo por rápida y media
    if EMA_SLOW > 0:
        long_stack  = close > ef > em > es
        short_stack = close < ef < em < es
    else:
        long_stack  = close > ef > em
        short_stack = close < ef < em

    # Filtro RSI
    if long_stack  and rsi_val < RSI_MIN:
        return None
    if short_stack and rsi_val > RSI_MAX:
        return None

    if long_stack:
        return 'long'
    if short_stack:
        return 'short'
    return None

def check_exit(df: pd.DataFrame, direction: str) -> bool:
    """Sale cuando el precio cruza la EMA media."""
    ema_mid = calc_ema(df['close'], EMA_MID)
    close   = df['close'].iloc[-2]
    em      = ema_mid.iloc[-2]
    if pd.isna(em):
        return False
    if direction == 'long'  and close < em:
        return True
    if direction == 'short' and close > em:
        return True
    return False

# ─── SIZING ───────────────────────────────────────────────────────────────────
def get_balance(client: Client) -> float:
    try:
        for b in client.futures_account_balance():
            if b['asset'] == 'USDT':
                return float(b['balance'])
    except Exception as e:
        log.error(f"Error balance: {e}")
    return 0.0

def get_step(client: Client, symbol: str) -> float:
    try:
        info = client.futures_exchange_info()
        for s in info['symbols']:
            if s['symbol'] == symbol:
                for f in s['filters']:
                    if f['filterType'] == 'LOT_SIZE':
                        return float(f['stepSize'])
    except Exception as e:
        log.error(f"Error step size: {e}")
    return 0.001

def calc_qty(balance: float, price: float, step: float) -> float:
    if price <= 0 or step <= 0:
        return 0.0
    notional = balance * LEVERAGE * 0.95  # 95% para dejar margen
    qty = notional / price
    qty = qty - (qty % step)
    return round(qty, 8)

# ─── EJECUCIÓN ────────────────────────────────────────────────────────────────
def open_position(client: Client, symbol: str, direction: str) -> dict | None:
    try:
        client.futures_change_leverage(symbol=symbol, leverage=LEVERAGE)
        price   = get_price(symbol)
        balance = get_balance(client)
        step    = get_step(client, symbol)
        qty     = calc_qty(balance, price, step)

        if qty <= 0:
            log.error(f"Qty=0 — balance: ${balance:.2f}, price: {price}")
            return None

        side  = SIDE_BUY if direction == 'long' else SIDE_SELL
        order = client.futures_create_order(
            symbol=symbol, side=side,
            type=ORDER_TYPE_MARKET, quantity=qty
        )

        # FIX: en testnet avgPrice puede venir como "0" — fallback al markPrice
        raw_fill = float(order.get('avgPrice') or 0)
        fill = raw_fill if raw_fill > 0 else price

        log.info(
            f"Abierto: {direction.upper()} {qty} {symbol} @ {fill:.2f} "
            f"| Balance: ${balance:.2f} | Lev: {LEVERAGE}×"
        )
        return {
            'symbol':     symbol,
            'direction':  direction,
            'qty':        qty,
            'entry':      fill,
            'balance_in': balance,
            'opened_at':  datetime.now(timezone.utc),
        }

    except Exception as e:
        log.error(f"Error abriendo posición: {e}")
        return None

def close_position(client: Client, trade: dict) -> float | None:
    try:
        side  = SIDE_SELL if trade['direction'] == 'long' else SIDE_BUY
        order = client.futures_create_order(
            symbol=trade['symbol'], side=side,
            type=ORDER_TYPE_MARKET,
            quantity=trade['qty'], reduceOnly=True
        )
        raw_price = float(order.get('avgPrice') or 0)
        price = raw_price if raw_price > 0 else get_price(trade['symbol'])
        log.info(f"Cerrado: {trade['symbol']} @ {price:.2f}")
        return price
    except Exception as e:
        log.error(f"Error cerrando: {e}")
        return None

# ─── HORARIO ──────────────────────────────────────────────────────────────────
def in_session(now: datetime) -> bool:
    """True durante sesión NY, sesión Asia, o fin de semana."""
    dow = now.weekday()   # 0=lun … 6=dom
    h   = now.hour
    m   = now.minute
    hm  = h * 60 + m

    # Fin de semana: operar siempre
    if dow >= 5:
        return True

    # Sesión NY: 13:30 – 20:00 UTC
    if 13 * 60 + 30 <= hm < 20 * 60:
        return True

    # Sesión Asia: 00:00 – 08:00 UTC
    if 0 <= hm < 8 * 60:
        return True

    return False

# ─── MENSAJES TELEGRAM ────────────────────────────────────────────────────────
def format_open(trade: dict) -> str:
    env = '🧪 TESTNET' if USE_TESTNET else '🔴 REAL'
    de  = '🟢 LONG' if trade['direction'] == 'long' else '🔴 SHORT'
    return (
        f"{'─'*28}\n⚡ *NUEVA POSICIÓN* {env}\n{'─'*28}\n"
        f"*Par:*      `{trade['symbol']}`\n"
        f"*Dir:*      {de}\n"
        f"*Entrada:*  `{trade['entry']:,.2f} USDT`\n"
        f"*Qty:*      `{trade['qty']}`\n"
        f"*Capital:*  `${trade['balance_in']:,.2f}`\n"
        f"*Leverage:* `{LEVERAGE}×`\n"
        f"*Exit:*     Cruce EMA{EMA_MID}\n"
        f"{'─'*28}"
    )

def format_close(trade: dict, exit_price: float) -> str:
    entry = trade.get('entry') or 0
    if entry > 0 and exit_price > 0:
        pnl = (exit_price - entry) / entry * 100
        if trade['direction'] == 'short':
            pnl = -pnl
        pnl_usd = trade['balance_in'] * pnl / 100 * LEVERAGE
        pnl_str = f"`{pnl:+.2f}%` (`{'+' if pnl_usd >= 0 else ''}{pnl_usd:.2f} USDT`)"
        re = '✅ WIN' if pnl > 0 else '❌ LOSS'
    else:
        pnl_str = "_no disponible_"
        re = '⚪ CERRADO'

    dur = datetime.now(timezone.utc) - trade['opened_at']
    h = int(dur.total_seconds() // 3600)
    m = int((dur.total_seconds() % 3600) // 60)
    de = '🟢 LONG' if trade['direction'] == 'long' else '🔴 SHORT'
    return (
        f"{'─'*28}\n🔔 *POSICIÓN CERRADA*\n{'─'*28}\n"
        f"*Resultado:* {re}\n"
        f"*Par:*       `{trade['symbol']}`\n"
        f"*Dir:*       {de}\n"
        f"*Entrada:*   `{entry:,.2f}`\n"
        f"*Salida:*    `{exit_price:,.2f}`\n"
        f"*P/L:*       {pnl_str}\n"
        f"*Duración:*  `{h}h {m}m`\n"
        f"{'─'*28}"
    )

async def send_tg(app: Application, text: str) -> None:
    try:
        await app.bot.send_message(
            chat_id=TELEGRAM_CHAT_ID, text=text, parse_mode='Markdown'
        )
    except Exception as e:
        log.error(f"Telegram error: {e}")

# ─── COMANDOS ─────────────────────────────────────────────────────────────────
async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global active_trade
    if not active_trade:
        await update.message.reply_text("📭 Sin posiciones abiertas.")
        return
    try:
        price = get_price(active_trade['symbol'])
        entry = active_trade.get('entry') or 0

        dur = datetime.now(timezone.utc) - active_trade['opened_at']
        h   = int(dur.total_seconds() // 3600)
        m   = int((dur.total_seconds() % 3600) // 60)
        dir_e = '🟢 LONG' if active_trade['direction'] == 'long' else '🔴 SHORT'

        # FIX: evitar division by zero si entry=0 o price=0
        if entry > 0 and price > 0:
            pnl = (price - entry) / entry * 100
            if active_trade['direction'] == 'short':
                pnl = -pnl
            pnl_str = f"`{pnl:+.2f}%`"
        else:
            pnl_str = "_entrada sin confirmar_"

        price_str = f"`{price:,.2f}`" if price > 0 else "_no disponible_"
        entry_str = f"`{entry:,.2f}`" if entry > 0 else "_pendiente de fill_"

        msg = (
            f"📊 *Posición activa*\n\n"
            f"*{active_trade['symbol']}* {dir_e}\n"
            f"Entrada: {entry_str}\n"
            f"Precio actual: {price_str}\n"
            f"P/L: {pnl_str}\n"
            f"Duración: `{h}h {m}m`"
        )
    except Exception as e:
        log.error(f"Error en /status: {e}")
        msg = f"⚠️ Error obteniendo status: {e}"

    await update.message.reply_text(msg, parse_mode='Markdown')

async def cmd_close(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global active_trade
    if not active_trade:
        await update.message.reply_text("📭 Sin posiciones abiertas.")
        return
    client = get_client()
    exit_price = close_position(client, active_trade)
    if exit_price:
        msg = format_close(active_trade, exit_price) + "\n_Cierre manual por comando._"
        active_trade = None
        await update.message.reply_text(msg, parse_mode='Markdown')
    else:
        await update.message.reply_text("❌ Error cerrando posición — verificá en Binance.")

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    env = '🧪 TESTNET' if USE_TESTNET else '🔴 REAL'
    msg = (
        f"🤖 *Brújula EMA Stack Bot* {env}\n\n"
        f"*Comandos:*\n"
        f"/status — estado de la posición activa\n"
        f"/close  — cerrar posición manualmente\n"
        f"/help   — este mensaje\n\n"
        f"*Config actual:*\n"
        f"Par: `{SYMBOL}` | Leverage: `{LEVERAGE}×`\n"
        f"EMAs: `{EMA_FAST}/{EMA_MID}/{EMA_SLOW}`\n"
        f"ADX: `{ADX_MIN}` | ATR: `{ATR_MIN_PCT}%`\n"
        f"RSI: `{RSI_MIN}/{RSI_MAX}` | Vol: `{VOL_MULT}×`\n"
        f"Scan: cada `{SCAN_INTERVAL}s`\n"
        f"Horario: NY + Asia + Fin de semana"
    )
    await update.message.reply_text(msg, parse_mode='Markdown')

# ─── SCAN ─────────────────────────────────────────────────────────────────────
async def scan_job(app: Application) -> None:
    global active_trade, last_candle_time
    log.info("Running scan...")

    try:
        client = get_client()
        df     = get_klines(SYMBOL)
    except Exception as e:
        log.error(f"Error obteniendo datos: {e}")
        return

    # ── Deduplicación por vela: una señal máximo por vela de 5m ──
    current_candle = int(df['open_time'].iloc[-2])
    if current_candle == last_candle_time:
        log.info(f"Vela ya evaluada ({current_candle}) — esperando nueva vela")
        return
    last_candle_time = current_candle

    # ── Gestión de posición abierta ──
    if active_trade:
        try:
            should_exit = check_exit(df, active_trade['direction'])
            if should_exit:
                exit_price = close_position(client, active_trade)
                if exit_price:
                    msg = format_close(active_trade, exit_price) + "\n_Cierre automático por EMA._"
                    active_trade = None
                    await send_tg(app, msg)
                    log.info("Posición cerrada por señal EMA")
                else:
                    log.error("Fallo cierre — reintento próximo scan")
        except Exception as e:
            log.error(f"Error exit check: {e}")
        return  # No abrir nuevas mientras hay posición

    # ── Verificar horario ──
    now = datetime.now(timezone.utc)
    if not in_session(now):
        log.info(f"Fuera de horario ({now.strftime('%H:%M UTC')} dow={now.weekday()})")
        return

    # ── Buscar señal de entrada ──
    try:
        signal = check_signal(df)
    except Exception as e:
        log.error(f"Error calculando señal: {e}")
        return

    if not signal:
        log.info("Sin señal")
        return

    log.info(f"Señal: {signal.upper()} — ejecutando...")
    trade = open_position(client, SYMBOL, signal)
    if trade:
        active_trade = trade
        await send_tg(app, format_open(trade))
    else:
        log.error("Fallo apertura de posición")

# ─── MAIN ─────────────────────────────────────────────────────────────────────
async def scan_callback(context: ContextTypes.DEFAULT_TYPE) -> None:
    await scan_job(context.application)

async def post_init(app: Application) -> None:
    app.job_queue.run_repeating(
        callback=scan_callback,
        interval=SCAN_INTERVAL,
        first=10,
        name='scan',
    )
    env = 'TESTNET 🧪' if USE_TESTNET else 'REAL 🔴'
    log.info(
        f"Bot iniciado — {SYMBOL} | {env} | Lev: {LEVERAGE}× | "
        f"EMAs: {EMA_FAST}/{EMA_MID}/{EMA_SLOW} | ADX: {ADX_MIN} | "
        f"ATR: {ATR_MIN_PCT}% | RSI: {RSI_MIN}/{RSI_MAX} | Vol: {VOL_MULT}×"
    )
    await app.bot.send_message(
        chat_id=TELEGRAM_CHAT_ID,
        parse_mode='Markdown',
        text=(
            f"🤖 *Brújula EMA Stack Bot iniciado* {'🧪 TESTNET' if USE_TESTNET else '🔴 REAL'}\n\n"
            f"*Par:* `{SYMBOL}` | *Leverage:* `{LEVERAGE}×`\n"
            f"*EMAs:* `{EMA_FAST}/{EMA_MID}/{EMA_SLOW}` | *ADX:* `{ADX_MIN}`\n"
            f"*ATR:* `{ATR_MIN_PCT}%` | *RSI:* `{RSI_MIN}/{RSI_MAX}` | *Vol:* `{VOL_MULT}×`\n"
            f"*Horario:* NY + Asia + Fin de semana\n\n"
            f"_Escaneando cada {SCAN_INTERVAL}s. /help para comandos._"
        )
    )

def main() -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        raise ValueError("TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID son requeridos")
    if not BINANCE_API_KEY or not BINANCE_SECRET:
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
