"""
Brújula Bot — ORB Combinado Londres + NY
=========================================
Modelo validado (2020-2026): Sharpe 1.70 · MaxDD -21.4%

SESIÓN LONDRES:
  Señal:   Primera vela 15m del día a las 8:00 GMT
  Entry:   Close de la vela ORB (o fallback hasta v4 si EMA falla)
  Stop:    Low (BULL) / High (BEAR) de la vela ORB
  Cierre:  11:00 GMT · EoD puro

SESIÓN NY:
  Señal:   Primera vela 15m del día a las 9:30 ET
  Entry:   Close de la vela ORB (o fallback hasta v4 si EMA falla)
  Stop:    Low (BULL) / High (BEAR) de la vela ORB
  Cierre:  15:00 ET · EoD puro

FILTROS (ambas sesiones):
  EMA50 15m  — LONG si close > EMA · SHORT si close < EMA
               Fallback: monitorea hasta 4 velas. Si ninguna cumple → descarta el día.
  Stop dist  — 0.2% < (entry-stop)/entry < 1.5%
               Filtra velas laterales (chicas) y explosivas (grandes).

CAPITAL: 98% del disponible en cada señal · compuesto trade a trade

Variables Railway:
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
  BINANCE_API_KEY, BINANCE_API_SECRET
  USE_TESTNET     (true)
  SYMBOL          (ETHUSDT)
  LEVERAGE        (3)
  CAPITAL_PCT     (98)
  EMA_PERIOD      (50)
  STOP_DIST_MIN   (0.2)
  STOP_DIST_MAX   (1.5)
  SCAN_INTERVAL   (30)
"""

import os, time, logging, math, hmac, hashlib, urllib.parse
import requests as req_lib
from datetime import datetime, timezone, timedelta

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
CAPITAL_PCT      = float(os.environ.get('CAPITAL_PCT', '98'))
EMA_PERIOD       = int(os.environ.get('EMA_PERIOD', '50'))
STOP_DIST_MIN    = float(os.environ.get('STOP_DIST_MIN', '0.2'))
STOP_DIST_MAX    = float(os.environ.get('STOP_DIST_MAX', '1.5'))
SCAN_INTERVAL    = int(os.environ.get('SCAN_INTERVAL', '30'))

# ─── ESTADO GLOBAL ────────────────────────────────────────────────────────────
active_trade:          dict | None = None
traded_london_today:   str  | None = None  # fecha ET 'YYYY-MM-DD'
traded_ny_today:       str  | None = None  # fecha ET 'YYYY-MM-DD'
last_15m_ts:           int  | None = None  # ts de la última vela 15m evaluada

# ─── HELPERS DE TIEMPO ────────────────────────────────────────────────────────
def is_edt(dt: datetime) -> bool:
    """EDT: segundo domingo de marzo → primer domingo de noviembre."""
    y = dt.year
    mar1 = datetime(y, 3, 1, tzinfo=timezone.utc)
    edt_start = mar1 + timedelta(days=(6 - mar1.weekday()) % 7 + 7)
    nov1 = datetime(y, 11, 1, tzinfo=timezone.utc)
    edt_end = nov1 + timedelta(days=(6 - nov1.weekday()) % 7)
    return edt_start <= dt.replace(tzinfo=timezone.utc) < edt_end

def is_bst(dt: datetime) -> bool:
    """BST: último domingo de marzo → último domingo de octubre."""
    y = dt.year
    mar = datetime(y, 3, 31, tzinfo=timezone.utc)
    while mar.weekday() != 6: mar -= timedelta(days=1)
    oct_ = datetime(y, 10, 31, tzinfo=timezone.utc)
    while oct_.weekday() != 6: oct_ -= timedelta(days=1)
    return mar <= dt.replace(tzinfo=timezone.utc) < oct_

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

def today_et() -> str:
    now = utc_now()
    offset = -4 if is_edt(now) else -5
    return (now + timedelta(hours=offset)).strftime('%Y-%m-%d')

def is_weekend() -> bool:
    now = utc_now()
    offset = -4 if is_edt(now) else -5
    return (now + timedelta(hours=offset)).weekday() >= 5

def utc_hour_now() -> float:
    d = utc_now()
    return d.hour + d.minute / 60 + d.second / 3600

# ─── DETECTAR VELAS ORB ───────────────────────────────────────────────────────
def is_orb_london(ts: int) -> bool:
    """8:00 GMT = 7:00 UTC (BST) / 8:00 UTC (GMT)."""
    d = datetime.fromtimestamp(ts, tz=timezone.utc)
    if d.minute != 0: return False
    return d.hour == 7 if is_bst(d) else d.hour == 8

def is_orb_ny(ts: int) -> bool:
    """9:30 ET = 13:30 UTC (EDT) / 14:30 UTC (EST)."""
    d = datetime.fromtimestamp(ts, tz=timezone.utc)
    if d.minute != 30: return False
    return d.hour == 13 if is_edt(d) else d.hour == 14

def close_utc_london() -> float:
    """11:00 GMT = 10:00 UTC (BST) / 11:00 UTC (GMT)."""
    now = utc_now()
    return 10.0 if is_bst(now) else 11.0

def close_utc_ny() -> float:
    """15:00 ET = 19:00 UTC (EDT) / 20:00 UTC (EST)."""
    now = utc_now()
    return 19.0 if is_edt(now) else 20.0

# ─── CLIENTE BINANCE ──────────────────────────────────────────────────────────
def get_client() -> Client:
    c = Client(BINANCE_KEY, BINANCE_SECRET, testnet=USE_TESTNET)
    if USE_TESTNET:
        c.FUTURES_URL = 'https://testnet.binancefuture.com/fapi'
    return c

def get_klines(symbol: str, interval: str, limit: int = 100) -> pd.DataFrame:
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

# ─── EMA ──────────────────────────────────────────────────────────────────────
def calc_ema(closes: list[float], period: int) -> list[float]:
    """EMA estándar sobre lista de closes. Seed con SMA de los primeros `period` valores."""
    if len(closes) < period:
        return [0.0] * len(closes)
    k = 2 / (period + 1)
    ema = [0.0] * len(closes)
    ema[period - 1] = sum(closes[:period]) / period
    for i in range(period, len(closes)):
        ema[i] = closes[i] * k + ema[i-1] * (1 - k)
    return ema

# ─── SEÑAL ORB ────────────────────────────────────────────────────────────────
def check_signal_orb(df15: pd.DataFrame, session: str) -> tuple[str|None, float, float, int]:
    """
    Evalúa la primera vela de la sesión y hasta 3 fallbacks (4 velas = 1h).
    Filtros aplicados sobre la vela seleccionada:
      1. EMA50 15m — LONG si close > EMA · SHORT si close < EMA
      2. Stop dist — STOP_DIST_MIN% < dist < STOP_DIST_MAX%
    Si ninguna vela cumple ambos filtros → sin señal.
    """
    is_orb_fn = is_orb_london if session == 'london' else is_orb_ny
    closed    = df15.iloc[:-1]   # excluir vela abierta actual
    hoy_et    = today_et()

    # Calcular EMA sobre todos los closes cerrados
    closes = [float(r['close']) for _, r in closed.iterrows()]
    ema_vals = calc_ema(closes, EMA_PERIOD) if EMA_PERIOD > 0 else [0.0] * len(closes)

    # Encontrar índice de la vela ORB de hoy
    orb_idx = None
    for idx, (_, row) in enumerate(closed.iterrows()):
        ts = int(row['open_time']) // 1000
        if not is_orb_fn(ts):
            continue
        orb_dt     = datetime.fromtimestamp(ts, tz=timezone.utc)
        orb_offset = -4 if is_edt(orb_dt) else -5
        orb_et     = (orb_dt + timedelta(hours=orb_offset)).strftime('%Y-%m-%d')
        if orb_et != hoy_et:
            return None, 0.0, 0.0, 0
        # Verificar que no sea demasiado antigua
        now_ts = int(utc_now().timestamp())
        orb_close_ts = ts + 15 * 60
        if (now_ts - orb_close_ts) / 60 > 30 + 3 * 15:  # hasta 4 velas (75 min)
            log.info(f"Vela ORB {session.upper()} demasiado antigua")
            return None, 0.0, 0.0, 0
        orb_idx = idx
        break

    if orb_idx is None:
        return None, 0.0, 0.0, 0

    # Probar vela ORB + hasta 3 fallbacks
    rows_list = list(closed.iterrows())
    for offset in range(4):
        ci = orb_idx + offset
        if ci >= len(rows_list):
            break
        _, row = rows_list[ci]
        o, h, l, c = float(row['open']), float(row['high']), float(row['low']), float(row['close'])

        # Doji exacto → skip
        if c == o:
            continue

        direction = 'long' if c > o else 'short'

        # Filtro EMA
        if EMA_PERIOD > 0 and ema_vals[ci] > 0:
            ema_ok = (direction == 'long' and c > ema_vals[ci]) or \
                     (direction == 'short' and c < ema_vals[ci])
            if not ema_ok:
                continue

        # Filtro stop dist
        sl_price  = l if direction == 'long' else h
        stop_dist = abs(c - sl_price) / c * 100
        if stop_dist < STOP_DIST_MIN or stop_dist > STOP_DIST_MAX:
            log.info(f"Stop dist {stop_dist:.3f}% fuera de rango [{STOP_DIST_MIN},{STOP_DIST_MAX}] — vela {offset+1}")
            continue

        ts = int(row['open_time']) // 1000
        log.info(f"Señal ORB {session.upper()} v{offset+1}: {direction.upper()} entry={c:.4f} sl={sl_price:.4f} dist={stop_dist:.3f}% ema={ema_vals[ci]:.4f}")
        return direction, c, sl_price, ts, offset + 1

    return None, 0.0, 0.0, 0, 0

# ─── ÓRDENES ──────────────────────────────────────────────────────────────────
def place_stop_order(symbol: str, direction: str, qty: float, stop_price: float) -> str | None:
    if USE_TESTNET:
        log.info(f"Testnet: stop software en {stop_price:.4f}")
        return None
    try:
        side   = "SELL" if direction == "long" else "BUY"
        ts_ms  = int(time.time() * 1000)
        params = {
            "symbol": symbol, "side": side, "type": "STOP_MARKET",
            "algoType": "CONDITIONAL", "quantity": str(qty),
            "triggerPrice": f"{stop_price:.2f}", "reduceOnly": "true",
            "workingType": "MARK_PRICE", "timestamp": ts_ms,
        }
        query = urllib.parse.urlencode(params)
        sig   = hmac.new(BINANCE_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
        params["signature"] = sig
        r = req_lib.post("https://fapi.binance.com/fapi/v1/algoOrder",
                         params=params, headers={"X-MBX-APIKEY": BINANCE_KEY}, timeout=10)
        data = r.json()
        if "algoId" in data:
            log.info(f"STOP_MARKET algo: {direction.upper()} stop={stop_price:.4f} id={data['algoId']}")
            return str(data["algoId"])
        log.error(f"Error algo stop: {data}")
        return None
    except Exception as e:
        log.error(f"Error colocando stop: {e}")
        return None

def cancel_stop_order(symbol: str, order_id: str | None) -> bool:
    if not order_id or USE_TESTNET:
        return True
    try:
        ts_ms  = int(time.time() * 1000)
        params = {"algoId": int(order_id), "timestamp": ts_ms}
        query  = urllib.parse.urlencode(params)
        sig    = hmac.new(BINANCE_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
        params["signature"] = sig
        r = req_lib.delete("https://fapi.binance.com/fapi/v1/algoOrder",
                           params=params, headers={"X-MBX-APIKEY": BINANCE_KEY}, timeout=10)
        log.info(f"Cancel stop {order_id}: {r.json()}")
        return True
    except Exception as e:
        log.error(f"Error cancelando stop: {e}")
        return False

def open_position(symbol: str, direction: str, sl_price: float, session: str, **kwargs) -> dict | None:
    try:
        balance  = get_balance()
        mark     = get_mark_price(symbol)
        step     = get_step_size(symbol)
        notional = balance * (CAPITAL_PCT / 100) * LEVERAGE
        qty      = round_qty(notional / mark, step)
        log.info(f"open_position: balance={balance:.2f} mark={mark:.4f} notional={notional:.2f} qty={qty} lev={LEVERAGE}x")

        if qty <= 0:
            log.error("Qty = 0, no abre")
            return None

        if direction == 'long'  and sl_price >= mark:
            log.error(f"SL {sl_price:.4f} >= mark {mark:.4f} — abortando")
            return None
        if direction == 'short' and sl_price <= mark:
            log.error(f"SL {sl_price:.4f} <= mark {mark:.4f} — abortando")
            return None

        try:
            get_client().futures_change_leverage(symbol=symbol, leverage=LEVERAGE)
        except Exception as e:
            log.warning(f"Leverage: {e}")

        side  = SIDE_BUY if direction == 'long' else SIDE_SELL
        order = get_client().futures_create_order(
            symbol=symbol, side=side, type=ORDER_TYPE_MARKET, quantity=qty
        )

        fills       = order.get('fills', [])
        entry_price = float(order.get('avgPrice', 0))
        if not entry_price and fills:
            total = sum(float(f['qty']) for f in fills)
            entry_price = sum(float(f['price'])*float(f['qty']) for f in fills) / total if total else mark
        if not entry_price:
            entry_price = mark

        stop_id = place_stop_order(symbol, direction, qty, sl_price)
        sl_pct  = abs(entry_price - sl_price) / entry_price * 100

        return {
            'symbol':        symbol,
            'session':       session,
            'direction':     direction,
            'qty':           qty,
            'entry':         entry_price,
            'sl_fixed':      sl_price,
            'sl_pct':        sl_pct,
            'best_swing':    entry_price,
            'trail_stop':    None,
            'stop_order_id': stop_id,
            'opened_at':     utc_now(),
            'orb_vela':      kwargs.get('orb_vela', 1),
        }
    except Exception as e:
        log.error(f"Error abriendo posición: {e}")
        return None

def close_position_market(trade: dict) -> float | None:
    try:
        cancel_stop_order(trade['symbol'], trade.get('stop_order_id'))
        side  = SIDE_SELL if trade['direction'] == 'long' else SIDE_BUY
        order = get_client().futures_create_order(
            symbol=trade['symbol'], side=side,
            type=ORDER_TYPE_MARKET, quantity=trade['qty'], reduceOnly=True
        )
        fills = order.get('fills', [])
        price = float(order.get('avgPrice', 0))
        if not price and fills:
            total = sum(float(f['qty']) for f in fills)
            price = sum(float(f['price'])*float(f['qty']) for f in fills) / total if total else None
        return price
    except Exception as e:
        log.error(f"Error cerrando: {e}")
        return None

# ─── TRAILING (solo NY) ───────────────────────────────────────────────────────
def update_trail_15m(trade: dict, df15: pd.DataFrame) -> bool:
    """Actualiza el trailing stop por closes de velas 15m cerradas. Solo para NY."""
    if trade.get('session') != 'ny':
        return False
    if TRAIL_PCT_NY <= 0:
        return False

    entry     = trade['entry']
    direction = trade['direction']
    opened_ts = int(trade['opened_at'].timestamp() * 1000)
    updated   = False

    closed = df15[df15['open_time'] < int(df15['open_time'].iloc[-1])]
    closed = closed[closed['open_time'] > opened_ts]

    for _, row in closed.iterrows():
        close_price = float(row['close'])
        if direction == 'long':
            if close_price > trade['best_swing']:
                trade['best_swing'] = close_price
                swing_dist  = trade['best_swing'] - entry
                new_trail   = entry + swing_dist * (TRAIL_PCT_NY / 100)
                if new_trail > (trade['trail_stop'] or trade['sl_fixed']):
                    trade['trail_stop'] = new_trail
                    cancel_stop_order(trade['symbol'], trade.get('stop_order_id'))
                    trade['stop_order_id'] = place_stop_order(
                        trade['symbol'], direction, trade['qty'], new_trail)
                    updated = True
        else:
            if close_price < trade['best_swing']:
                trade['best_swing'] = close_price
                swing_dist  = entry - trade['best_swing']
                new_trail   = entry - swing_dist * (TRAIL_PCT_NY / 100)
                if new_trail < (trade['trail_stop'] or trade['sl_fixed']):
                    trade['trail_stop'] = new_trail
                    cancel_stop_order(trade['symbol'], trade.get('stop_order_id'))
                    trade['stop_order_id'] = place_stop_order(
                        trade['symbol'], direction, trade['qty'], new_trail)
                    updated = True
    return updated

# ─── MENSAJES TELEGRAM ────────────────────────────────────────────────────────
async def send_tg(app: Application, text: str) -> None:
    try:
        await app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text)
    except Exception as e:
        log.error(f"Telegram: {e}")

def fmt_open(trade: dict, balance: float) -> str:
    env  = '🧪 TESTNET' if USE_TESTNET else '🔴 REAL'
    icon = '🇬🇧' if trade['session'] == 'london' else '🇺🇸'
    sess = 'LONDRES' if trade['session'] == 'london' else 'NY'
    close_info = '11:00 GMT' if trade['session'] == 'london' else '15:00 ET'
    orb_vela = trade.get('orb_vela', 1)
    vela_str = f" (v{orb_vela})" if orb_vela > 1 else ""
    return (
        f"{'─'*28}\n"
        f"⚡ ENTRADA ORB {icon} {sess} {env}\n"
        f"{'─'*28}\n"
        f"Par:      {trade['symbol']}\n"
        f"Dir:      {'🟢 LONG' if trade['direction']=='long' else '🔴 SHORT'}{vela_str}\n"
        f"Entry:    {trade['entry']:,.4f}\n"
        f"SL ORB:   {trade['sl_fixed']:,.4f} (-{trade['sl_pct']:.2f}%)\n"
        f"Cierre:   {close_info} · EoD\n"
        f"Capital:  ${balance:,.2f} × {LEVERAGE}x\n"
        f"{'─'*28}"
    )

def fmt_close(trade: dict, exit_price: float, reason: str) -> str:
    pnl_pct = (exit_price - trade['entry']) * (1 if trade['direction']=='long' else -1) / trade['entry'] * 100
    pnl_lev = pnl_pct * LEVERAGE
    icon = '🇬🇧' if trade['session'] == 'london' else '🇺🇸'
    sess = 'LONDRES' if trade['session'] == 'london' else 'NY'
    reasons = {'stop':'🛑 Stop','trailing':'🔄 Trailing','close_time':'⏰ Tiempo','manual':'✋ Manual'}
    return (
        f"{'─'*28}\n"
        f"{reasons.get(reason,'📤')} CIERRE {icon} {sess}\n"
        f"{'─'*28}\n"
        f"Dir:      {'🟢 LONG' if trade['direction']=='long' else '🔴 SHORT'}\n"
        f"Entry:    {trade['entry']:,.4f}\n"
        f"Exit:     {exit_price:,.4f}\n"
        f"PnL:      {pnl_pct:>+.3f}% (precio)\n"
        f"PnL:      {pnl_lev:>+.3f}% ({LEVERAGE}x capital)\n"
        f"{'─'*28}"
    )

# ─── COMANDOS TELEGRAM ────────────────────────────────────────────────────────
async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    env = '🧪 TESTNET' if USE_TESTNET else '🔴 REAL'
    await update.message.reply_text(
        f"🤖 Brújula Bot ORB Combinado {env}\n\n"
        f"🇬🇧 Londres: 8:00 GMT → 11:00 GMT · EoD\n"
        f"🇺🇸 NY: 9:30 ET → 15:00 ET · EoD\n"
        f"📊 EMA{EMA_PERIOD} · Dist {STOP_DIST_MIN}%–{STOP_DIST_MAX}%\n"
        f"Par: {SYMBOL} · Lev: {LEVERAGE}x · Capital: {CAPITAL_PCT}%\n\n"
        f"/status /close /balance /help"
    )

async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "/start   — info del bot\n"
        "/status  — posición activa\n"
        "/close   — cerrar posición manualmente\n"
        "/balance — balance USDT\n"
        "/help    — esta ayuda"
    )

async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not active_trade:
        hoy = today_et()
        lon = '✅' if traded_london_today == hoy else '⏳'
        ny  = '✅' if traded_ny_today == hoy else '⏳'
        await update.message.reply_text(
            f"📭 Sin posición activa\n"
            f"🇬🇧 Londres hoy: {lon}\n"
            f"🇺🇸 NY hoy: {ny}"
        )
        return
    t = active_trade
    stop_activo = t.get('trail_stop') or t['sl_fixed']
    sess = '🇬🇧 LONDRES' if t['session']=='london' else '🇺🇸 NY'
    try:
        mark = get_mark_price(t['symbol'])
        pnl  = (mark - t['entry']) * (1 if t['direction']=='long' else -1) / t['entry'] * 100
    except:
        mark = pnl = 0
    await update.message.reply_text(
        f"📊 Posición activa — {sess}\n"
        f"Dir:   {'🟢 LONG' if t['direction']=='long' else '🔴 SHORT'}\n"
        f"Entry: {t['entry']:,.4f}\n"
        f"Mark:  {mark:,.4f}\n"
        f"PnL:   {pnl:>+.3f}%\n"
        f"Stop:  {stop_activo:,.4f}\n"
        f"Trail: {'activo' if t.get('trail_stop') else 'pendiente'}"
    )

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
    await update.message.reply_text(f"💰 Balance: ${bal:,.2f} USDT")

# ─── SCAN PRINCIPAL ───────────────────────────────────────────────────────────
async def scan(app: Application) -> None:
    global active_trade, traded_london_today, traded_ny_today, last_15m_ts

    now_utc  = utc_now()
    utc_h    = utc_hour_now()
    hoy      = today_et()

    # ── FILTRO FIN DE SEMANA
    if is_weekend():
        return

    # ── 1. GESTIONAR TRADE ABIERTO ───────────────────────────────────────────
    if active_trade:
        session = active_trade['session']

        # 1a. Cierre forzado por tiempo
        close_h = close_utc_london() if session == 'london' else close_utc_ny()
        if utc_h >= close_h:
            log.info(f"Cierre por tiempo — {session.upper()}")
            exit_price = close_position_market(active_trade)
            if exit_price is None:
                try: exit_price = get_mark_price(active_trade['symbol'])
                except: exit_price = active_trade['entry']
            msg = fmt_close(active_trade, exit_price, 'close_time')
            active_trade = None
            await send_tg(app, msg)
            return

        # 1b. Stop software (respaldo si no hay orden en Binance)
        if not active_trade.get('stop_order_id'):
            try:
                mark        = get_mark_price(active_trade['symbol'])
                stop_activo = active_trade.get('trail_stop') or active_trade['sl_fixed']
                direction   = active_trade['direction']
                tocado = (direction=='long' and mark<=stop_activo) or \
                         (direction=='short' and mark>=stop_activo)
                if tocado:
                    reason = 'trailing' if active_trade.get('trail_stop') else 'stop'
                    exit_price = close_position_market(active_trade)
                    if exit_price is None: exit_price = mark
                    msg = fmt_close(active_trade, exit_price, reason)
                    active_trade = None
                    await send_tg(app, msg)
                    return
            except Exception as e:
                log.error(f"Error stop software: {e}")

        # 1c. ¿Binance cerró la posición?
        try:
            positions = get_client().futures_position_information(symbol=SYMBOL)
            pos_amt   = float(positions[0]['positionAmt']) if positions else 0.0
            if abs(pos_amt) < 0.001:
                reason = 'trailing' if active_trade.get('trail_stop') else 'stop'
                exit_p = active_trade.get('trail_stop') or active_trade['sl_fixed']
                msg    = fmt_close(active_trade, exit_p, reason)
                active_trade = None
                await send_tg(app, msg)
                return
        except Exception as e:
            log.error(f"Error verificando posición: {e}")

        # 1d. Log estado del trade activo
        stop_activo = active_trade.get('trail_stop') or active_trade['sl_fixed']
        log.info(f"{session.upper()} {active_trade['direction'].upper()} entry={active_trade['entry']:.4f} stop={stop_activo:.4f}")
        return

    # ── 2. SIN TRADE — BUSCAR SEÑALES ────────────────────────────────────────
    try:
        df15 = get_klines(SYMBOL, '15m', limit=100)

        # Anti-duplicado por vela
        current_ts = int(df15['open_time'].iloc[-2])
        if current_ts == last_15m_ts:
            return
        last_15m_ts = current_ts

        # ── 2a. SEÑAL LONDRES (8:00 GMT, cierra 11:00 GMT)
        lon_close_utc = close_utc_london()
        if utc_h < lon_close_utc and traded_london_today != hoy:
            open_utc_lon = 7.0 if is_bst(now_utc) else 8.0
            if utc_h >= open_utc_lon:
                direction, entry, sl_price, orb_ts, orb_vela = check_signal_orb(df15, 'london')
                if direction and orb_ts:
                    orb_date    = datetime.fromtimestamp(orb_ts, tz=timezone.utc)
                    orb_offset  = -4 if is_edt(orb_date) else -5
                    orb_et_date = (orb_date + timedelta(hours=orb_offset)).strftime('%Y-%m-%d')
                    if orb_et_date == hoy:
                        sl_pct = abs(entry - sl_price) / entry * 100
                        log.info(f"Señal ORB LONDRES v{orb_vela}: {direction.upper()} entry={entry:.4f} sl={sl_price:.4f} (-{sl_pct:.2f}%)")
                        bal_pre = get_balance()
                        trade = open_position(SYMBOL, direction, sl_price, 'london', orb_vela=orb_vela)
                        if trade:
                            active_trade        = trade
                            traded_london_today = hoy
                            await send_tg(app, fmt_open(trade, bal_pre))
                        else:
                            traded_london_today = hoy

        # ── 2b. SEÑAL NY (9:30 ET, cierra 15:00 ET · EoD puro)
        ny_open_utc  = 13.5 if is_edt(now_utc) else 14.5
        ny_close_utc = close_utc_ny()
        if utc_h >= ny_open_utc and utc_h < ny_close_utc and traded_ny_today != hoy:
            direction, entry, sl_price, orb_ts, orb_vela = check_signal_orb(df15, 'ny')
            if direction and orb_ts:
                orb_date    = datetime.fromtimestamp(orb_ts, tz=timezone.utc)
                orb_offset  = -4 if is_edt(orb_date) else -5
                orb_et_date = (orb_date + timedelta(hours=orb_offset)).strftime('%Y-%m-%d')
                if orb_et_date == hoy:
                    sl_pct = abs(entry - sl_price) / entry * 100
                    log.info(f"Señal ORB NY v{orb_vela}: {direction.upper()} entry={entry:.4f} sl={sl_price:.4f} (-{sl_pct:.2f}%)")
                    bal_pre = get_balance()
                    trade = open_position(SYMBOL, direction, sl_price, 'ny', orb_vela=orb_vela)
                    if trade:
                        active_trade    = trade
                        traded_ny_today = hoy
                        await send_tg(app, fmt_open(trade, bal_pre))
                    else:
                        traded_ny_today = hoy

    except Exception as e:
        log.error(f"Error en scan: {e}")

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main() -> None:
    env = '🧪 TESTNET' if USE_TESTNET else '🔴 REAL'
    log.info(f"Brújula Bot ORB Combinado arrancando — {env}")

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
            ('start',   'Info del bot'),
            ('status',  'Posición activa'),
            ('close',   'Cerrar posición'),
            ('balance', 'Balance USDT'),
            ('help',    'Ayuda'),
        ])
        try:
            await app.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=(
                    f"🤖 Brújula Bot ORB Combinado {env}\n\n"
                    f"🇬🇧 Londres: 8:00 GMT → 11:00 GMT · EoD\n"
                    f"🇺🇸 NY: 9:30 ET → 15:00 ET · EoD\n"
                    f"📊 EMA{EMA_PERIOD} · Dist {STOP_DIST_MIN}%–{STOP_DIST_MAX}%\n"
                    f"Par: {SYMBOL} · Lev: {LEVERAGE}x · Capital: {CAPITAL_PCT}%\n"
                    f"Scan: cada {SCAN_INTERVAL}s · Sin solapamiento de sesiones"
                )
            )
        except Exception as e:
            log.error(f"Error mensaje inicio: {e}")

        loop = asyncio.get_event_loop()
        scheduler = AsyncIOScheduler()
        scheduler.add_job(
            lambda: loop.create_task(scan(app)),
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
