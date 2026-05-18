"""
Brújula Trading Bot v2 — EMA5 + ADX20 + Trailing 80%
=======================================================
Modelo validado contra tester v7e:
  Señal:    4h  — EMA5 + ADX20 (sesgo de dirección)
  Entrada:  15m — primera vela verde (long) o roja (short) → entra al CLOSE
  SL fijo:  0.5% desde la entrada → STOP_MARKET inmediata en Binance
  Trailing: swing crece con CLOSE de velas de 1h ya cerradas
            trail = entrada + swing × 80% (long) / entrada - swing × 80% (short)
            cuando trail supera SL fijo → reemplaza la STOP_MARKET
  Gatillo:  la STOP_MARKET de Binance ejecuta al nivel exacto (mark price)
  Reentrada: NO implementada (complejidad operativa — una posición a la vez)

Variables Railway:
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
  BINANCE_API_KEY, BINANCE_API_SECRET
  USE_TESTNET       (true)
  TRADING_SYMBOL    (ETHUSDT)
  LEVERAGE          (2)
  CAPITAL_PCT       (95)
  EMA_PERIOD        (5)
  ADX_PERIOD        (14)
  ADX_MIN           (20)
  SL_PCT            (0.5)
  TRAIL_PCT         (80)
  SCAN_INTERVAL     (60)   ← cada 1 minuto para detectar nuevas velas de 1h
"""

import os, time, logging, math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from binance.client import Client
from binance.enums import SIDE_BUY, SIDE_SELL, ORDER_TYPE_MARKET, FUTURE_ORDER_TYPE_STOP_MARKET
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN    = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID  = os.environ.get('TELEGRAM_CHAT_ID', '')
BINANCE_KEY       = os.environ.get('BINANCE_API_KEY', '')
BINANCE_SECRET    = os.environ.get('BINANCE_API_SECRET', '')
USE_TESTNET       = os.environ.get('USE_TESTNET', 'true').lower() == 'true'
SYMBOL            = os.environ.get('TRADING_SYMBOL', 'ETHUSDT')
LEVERAGE          = int(os.environ.get('LEVERAGE', '2'))
CAPITAL_PCT       = float(os.environ.get('CAPITAL_PCT', '95'))
EMA_PERIOD        = int(os.environ.get('EMA_PERIOD', '5'))
ADX_PERIOD        = int(os.environ.get('ADX_PERIOD', '14'))
ADX_MIN           = float(os.environ.get('ADX_MIN', '20'))
SL_PCT            = float(os.environ.get('SL_PCT', '0.5'))
TRAIL_PCT         = float(os.environ.get('TRAIL_PCT', '80'))
SCAN_INTERVAL     = int(os.environ.get('SCAN_INTERVAL', '60'))

MAX_ENTRY_CANDLES = 16   # máx velas 15m para confirmar entrada (= 4h)

# ─── ESTADO GLOBAL ────────────────────────────────────────────────────────────
active_trade:         dict | None = None
pending_signal:       str  | None = None   # 'long' o 'short'
pending_signal_ts:    int  | None = None   # timestamp cierre vela 4h señal (segundos Unix)
last_4h_candle:       int  | None = None   # open_time de la última vela 4h evaluada
last_1h_candle:       int  | None = None   # open_time de la última vela 1h procesada

# ─── CLIENTE BINANCE ──────────────────────────────────────────────────────────
def get_client() -> Client:
    c = Client(BINANCE_KEY, BINANCE_SECRET, testnet=USE_TESTNET)
    if USE_TESTNET:
        c.FUTURES_URL = 'https://testnet.binancefuture.com/fapi'
    return c

def get_klines(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    raw = get_client().futures_klines(symbol=symbol, interval=interval, limit=limit)
    df  = pd.DataFrame(raw, columns=[
        'open_time','open','high','low','close','vol',
        'close_time','qvol','trades','tbb','tbq','ignore'
    ])
    for col in ['open','high','low','close','vol']:
        df[col] = pd.to_numeric(df[col])
    df['open_time']  = df['open_time'].astype(int)
    df['close_time'] = df['close_time'].astype(int)
    return df

def get_mark_price(symbol: str) -> float:
    try:
        return float(get_client().futures_mark_price(symbol=symbol)['markPrice'])
    except Exception as e:
        log.error(f"Error mark price: {e}"); return 0.0

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
        for s in get_client().futures_exchange_info()['symbols']:
            if s['symbol'] == symbol:
                for f in s['filters']:
                    if f['filterType'] == 'LOT_SIZE':
                        return float(f['stepSize'])
    except Exception as e:
        log.error(f"Error step size: {e}")
    return 0.001

def get_tick_size(symbol: str) -> float:
    try:
        for s in get_client().futures_exchange_info()['symbols']:
            if s['symbol'] == symbol:
                for f in s['filters']:
                    if f['filterType'] == 'PRICE_FILTER':
                        return float(f['tickSize'])
    except Exception as e:
        log.error(f"Error tick size: {e}")
    return 0.01

def round_price(price: float, tick: float) -> float:
    decimals = max(0, round(-math.log10(tick)))
    return round(round(price / tick) * tick, decimals)

def get_position_amt(symbol: str) -> float:
    try:
        for pos in get_client().futures_position_information(symbol=symbol):
            if pos['symbol'] == symbol:
                return float(pos['positionAmt'])
    except Exception as e:
        log.error(f"Error position: {e}")
    return 0.0

def position_is_open(symbol: str) -> bool:
    return abs(get_position_amt(symbol)) > 0

# ─── ÓRDENES STOP ─────────────────────────────────────────────────────────────
def place_stop_order(symbol: str, direction: str, qty: float, stop_price: float) -> str | None:
    try:
        client = get_client()
        tick   = get_tick_size(symbol)
        sp     = round_price(stop_price, tick)
        side   = SIDE_SELL if direction == 'long' else SIDE_BUY
        decimals = max(0, round(-math.log10(tick)))
        order  = client.futures_create_order(
            symbol      = symbol,
            side        = side,
            type        = FUTURE_ORDER_TYPE_STOP_MARKET,
            stopPrice   = f"{sp:.{decimals}f}",
            quantity    = qty,
            reduceOnly  = True,
            workingType = 'MARK_PRICE',
        )
        oid = str(order['orderId'])
        log.info(f"STOP_MARKET colocada: {direction.upper()} stop={sp:.4f} id={oid}")
        return oid
    except Exception as e:
        log.error(f"Error STOP_MARKET: {e}"); return None

def cancel_stop_order(symbol: str, order_id: str | None) -> bool:
    if not order_id:
        return True
    try:
        get_client().futures_cancel_order(symbol=symbol, orderId=int(order_id))
        log.info(f"Orden {order_id} cancelada")
        return True
    except Exception as e:
        log.warning(f"Cancel {order_id}: {e}"); return False

# ─── INDICADORES ──────────────────────────────────────────────────────────────
def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def calc_adx(df: pd.DataFrame, period: int) -> pd.Series:
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

    def wilder(s: pd.Series, n: int) -> pd.Series:
        r = np.full(len(s), np.nan)
        r[n] = s.iloc[1:n+1].sum()
        for i in range(n + 1, len(s)):
            r[i] = r[i-1] - r[i-1] / n + s.iloc[i]
        return pd.Series(r, index=s.index)

    tr_w  = wilder(tr, period)
    pdm_w = wilder(pd.Series(pdm, index=df.index), period)
    ndm_w = wilder(pd.Series(ndm, index=df.index), period)
    pdi   = (pdm_w / tr_w * 100).replace([np.inf, -np.inf], np.nan)
    ndi   = (ndm_w / tr_w * 100).replace([np.inf, -np.inf], np.nan)
    dx    = ((pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan) * 100)
    return wilder(dx.fillna(0), period) / period

# ─── SEÑAL EN 4H ──────────────────────────────────────────────────────────────
def check_signal_4h(df4h: pd.DataFrame) -> tuple[str | None, int]:
    """
    Evalúa la penúltima vela de 4h (última cerrada).
    Retorna (dirección, timestamp_cierre_vela) o (None, 0).
    timestamp_cierre = open_time + 14400s
    """
    if len(df4h) < max(EMA_PERIOD, ADX_PERIOD * 3) + 10:
        return None, 0

    df    = df4h.iloc[:-1].copy()   # excluir vela abierta
    ema   = calc_ema(df['close'], EMA_PERIOD)
    adx   = calc_adx(df, ADX_PERIOD)

    close = float(df['close'].iloc[-1])
    ema_v = float(ema.iloc[-1])
    adx_v = float(adx.iloc[-1])

    # timestamp de cierre = open_time (ms) / 1000 + 14400s
    ts_close = int(df['open_time'].iloc[-1]) // 1000 + 14400

    if any(pd.isna(x) for x in [ema_v, adx_v]) or close <= 0:
        return None, 0

    log.info(f"4h | close={close:.2f} EMA{EMA_PERIOD}={ema_v:.2f} ADX={adx_v:.2f}/{ADX_MIN}")

    if adx_v < ADX_MIN:
        log.info(f"Filtrado ADX {adx_v:.2f} < {ADX_MIN}")
        return None, 0

    if close > ema_v: return 'long',  ts_close
    if close < ema_v: return 'short', ts_close
    return None, 0

# ─── CONFIRMACIÓN EN 15M ──────────────────────────────────────────────────────
def find_entry_15m(direction: str, signal_ts: int,
                   df15: pd.DataFrame) -> tuple[bool, float, bool]:
    """
    Busca la primera vela 15m verde (long) o roja (short) cuyo open_time
    es >= signal_ts (cierre de la vela 4h de señal).
    Entra al CLOSE de esa vela — replica exactamente el tester v7e.
    Máximo MAX_ENTRY_CANDLES velas (excluye la vela abierta).
    Retorna (confirmado, precio_entrada, expirado).
    """
    # Velas cerradas desde el cierre de la señal (excluir última, puede estar abierta)
    mask = (df15['open_time'] // 1000 >= signal_ts)
    post = df15[mask].iloc[:-1].reset_index(drop=True)

    if len(post) == 0:
        return False, 0.0, False

    for i, row in post.iterrows():
        op = float(row['open'])
        cl = float(row['close'])
        ts = int(row['open_time']) // 1000

        verde = cl > op
        roja  = cl < op

        if direction == 'long'  and verde:
            log.info(f"Confirmación 15m LONG en vela {i+1}: close={cl:.4f}")
            return True, cl, False

        if direction == 'short' and roja:
            log.info(f"Confirmación 15m SHORT en vela {i+1}: close={cl:.4f}")
            return True, cl, False

        log.info(f"Vela 15m {i+1}/{MAX_ENTRY_CANDLES}: {'verde' if verde else 'roja' if roja else 'doji'} — no confirma para {direction}")

        if i + 1 >= MAX_ENTRY_CANDLES:
            log.info(f"Señal expiró tras {MAX_ENTRY_CANDLES} velas sin confirmar")
            return False, 0.0, True

    return False, 0.0, False

# ─── TRAILING CON VELAS DE 1H ─────────────────────────────────────────────────
def update_trail_stop_1h(trade: dict, df1h: pd.DataFrame) -> bool:
    """
    Actualiza el trailing usando el close de la última vela de 1h CERRADA
    después de la entrada del trade. Replica la lógica del tester v7e.

    El swing solo crece (nunca retrocede).
    El trail toma control cuando supera al SL fijo.
    Si mejora, cancela la STOP_MARKET anterior y coloca una nueva.

    Retorna True si se actualizó la orden stop.
    """
    direction = trade['direction']
    entry_ts  = trade['entry_ts']   # timestamp Unix del open de la vela 15m de entrada
    entry     = trade['entry']
    sl_fixed  = trade['sl_fixed']

    # Solo usar velas de 1h que:
    # a) abrieron DESPUÉS de la entrada (entry_ts)
    # b) ya cerraron (close_time < ahora en ms)
    now_ms = int(time.time() * 1000)
    df_c   = df1h[
        (df1h['open_time'] // 1000 >= entry_ts) &
        (df1h['close_time'] < now_ms)
    ].copy()

    if df_c.empty:
        log.info("Trail 1h: sin velas cerradas desde la entrada aún")
        return False

    # Actualizar best_swing con todos los closes de 1h disponibles
    for _, row in df_c.iterrows():
        close_1h = float(row['close'])
        if direction == 'long'  and close_1h > trade['best_swing']:
            trade['best_swing'] = close_1h
            log.info(f"Trail 1h: nuevo máximo swing={close_1h:.4f}")
        elif direction == 'short' and close_1h < trade['best_swing']:
            trade['best_swing'] = close_1h
            log.info(f"Trail 1h: nuevo mínimo swing={close_1h:.4f}")

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

    if trail_stop is None:
        log.info(f"Trail 1h: swing insuficiente (best={best_swing:.4f} entry={entry:.4f})")
        return False

    # El trail toma control solo cuando supera al SL fijo
    trail_activo = (
        (direction == 'long'  and trail_stop >= sl_fixed) or
        (direction == 'short' and trail_stop <= sl_fixed)
    )

    if not trail_activo:
        log.info(f"Trail 1h: {trail_stop:.4f} aún no supera SL fijo {sl_fixed:.4f}")
        return False

    # Solo actualizar si el trail mejoró respecto al anterior
    prev = trade.get('trail_stop')
    mejoro = (
        prev is None or
        (direction == 'long'  and trail_stop > prev) or
        (direction == 'short' and trail_stop < prev)
    )

    if not mejoro:
        log.info(f"Trail 1h: sin mejora ({trail_stop:.4f} vs prev={prev:.4f})")
        return False

    log.info(f"Trail 1h MEJORÓ: {prev} → {trail_stop:.4f} | Actualizando STOP_MARKET")

    # Cancelar orden anterior y colocar nueva al nivel del trailing
    cancel_stop_order(trade['symbol'], trade.get('stop_order_id'))
    new_id = place_stop_order(trade['symbol'], direction, trade['qty'], trail_stop)

    trade['trail_stop']    = trail_stop
    trade['active_stop']   = trail_stop
    trade['stop_order_id'] = new_id
    return True

# ─── ABRIR POSICIÓN ───────────────────────────────────────────────────────────
def open_position(direction: str, entry_price: float, entry_ts: int) -> dict | None:
    """
    Abre posición market y coloca STOP_MARKET al SL fijo inmediatamente.
    entry_ts: timestamp Unix del open de la vela 15m confirmadora.
    """
    try:
        client = get_client()

        # Margen cruzado
        try:
            client.futures_change_margin_type(symbol=SYMBOL, marginType='CROSSED')
        except Exception:
            pass

        # Leverage
        lev_usado = LEVERAGE
        for lev in [LEVERAGE, LEVERAGE - 1, 1]:
            try:
                client.futures_change_leverage(symbol=SYMBOL, leverage=lev)
                lev_usado = lev
                break
            except Exception as e:
                log.warning(f"Leverage {lev}x rechazado: {e}")

        balance = get_balance()
        price   = entry_price if entry_price > 0 else get_mark_price(SYMBOL)
        step    = get_step_size(SYMBOL)

        if price <= 0 or balance <= 0:
            log.error(f"Datos inválidos: price={price} balance={balance}")
            return None

        notional = balance * (CAPITAL_PCT / 100) * lev_usado
        qty      = notional / price
        qty      = qty - (qty % step)
        qty      = round(qty, 8)

        if qty <= 0:
            log.error(f"qty=0 — balance={balance} price={price}")
            return None

        # Market order
        side  = SIDE_BUY if direction == 'long' else SIDE_SELL
        order = client.futures_create_order(
            symbol=SYMBOL, side=side,
            type=ORDER_TYPE_MARKET, quantity=qty
        )

        # Obtener precio real de entrada
        time.sleep(1)
        entry = float(order.get('avgPrice') or 0)
        try:
            for pos in client.futures_position_information(symbol=SYMBOL):
                if pos['symbol'] == SYMBOL and abs(float(pos['positionAmt'])) > 0:
                    ep = float(pos['entryPrice'])
                    if ep > 0:
                        entry = ep
                        break
        except Exception:
            pass
        if entry <= 0:
            entry = price

        # SL fijo
        sl_fixed = (
            entry * (1 - SL_PCT / 100) if direction == 'long'
            else entry * (1 + SL_PCT / 100)
        )

        # STOP_MARKET inmediata al SL fijo
        stop_order_id = place_stop_order(SYMBOL, direction, qty, sl_fixed)

        log.info(
            f"ABIERTO: {direction.upper()} {qty} {SYMBOL} @ {entry:.4f} "
            f"SL={sl_fixed:.4f} stop_id={stop_order_id}"
        )

        return {
            'symbol':        SYMBOL,
            'direction':     direction,
            'qty':           qty,
            'entry':         entry,
            'entry_ts':      entry_ts,   # para filtrar velas 1h desde la entrada
            'sl_fixed':      sl_fixed,
            'best_swing':    entry,       # swing arranca desde la entrada
            'trail_stop':    None,
            'active_stop':   sl_fixed,
            'stop_order_id': stop_order_id,
            'balance_in':    balance,
            'leverage':      lev_usado,
            'opened_at':     datetime.now(timezone.utc),
        }
    except Exception as e:
        log.error(f"Error abriendo posición: {e}")
        return None

# ─── CERRAR POSICIÓN ──────────────────────────────────────────────────────────
def close_position_market(trade: dict) -> float | None:
    """Cierre manual por market order. Cancela stop pendiente."""
    try:
        cancel_stop_order(trade['symbol'], trade.get('stop_order_id'))
        side  = SIDE_SELL if trade['direction'] == 'long' else SIDE_BUY
        order = get_client().futures_create_order(
            symbol=trade['symbol'], side=side,
            type=ORDER_TYPE_MARKET, quantity=trade['qty'], reduceOnly=True
        )
        price = float(order.get('avgPrice') or 0) or get_mark_price(trade['symbol'])
        log.info(f"Cerrado manual @ {price:.4f}")
        return price
    except Exception as e:
        log.error(f"Error cerrando: {e}"); return None

# ─── MENSAJES TELEGRAM ────────────────────────────────────────────────────────
def fmt_open(trade: dict) -> str:
    env   = '🧪 TESTNET' if USE_TESTNET else '🔴 REAL'
    emoji = '🟢' if trade['direction'] == 'long' else '🔴'
    return (
        f"{'─'*30}\n⚡ *ENTRADA* {env}\n{'─'*30}\n"
        f"*Par:*      `{trade['symbol']}`\n"
        f"*Dir:*      {emoji} `{trade['direction'].upper()}`\n"
        f"*Precio:*   `{trade['entry']:,.4f}`\n"
        f"*Qty:*      `{trade['qty']}`\n"
        f"*SL fijo:*  `{trade['sl_fixed']:,.4f}` (-{SL_PCT}%)\n"
        f"*Trail:*    `{TRAIL_PCT}%` del swing — actualiza por cierre 1h\n"
        f"*Stop ID:*  `{trade.get('stop_order_id', 'N/A')}`\n"
        f"*Capital:*  `${trade['balance_in']:,.2f}` × {trade['leverage']}×\n"
        f"{'─'*30}"
    )

def fmt_close(trade: dict, exit_price: float, reason: str) -> str:
    entry   = trade['entry']
    dir_    = trade['direction']
    pnl_pct = (
        (exit_price - entry) / entry * 100 if dir_ == 'long'
        else (entry - exit_price) / entry * 100
    )
    pnl_usdt = (
        trade['balance_in'] * (CAPITAL_PCT / 100) *
        trade.get('leverage', LEVERAGE) * pnl_pct / 100
    )
    dur    = datetime.now(timezone.utc) - trade['opened_at']
    h, m   = divmod(int(dur.total_seconds()), 3600)
    m      = m // 60
    result = '✅ WIN' if pnl_pct > 0 else '❌ LOSS'
    reason_str = {
        'sl':       '🛑 Stop Loss fijo (Binance STOP_MARKET)',
        'trailing': '📍 Trailing stop (Binance STOP_MARKET)',
        'ema':      '🟣 Cruce EMA (market)',
        'manual':   '🖐 Cierre manual',
    }.get(reason, reason)
    trail_str = f"`{trade['trail_stop']:,.4f}`" if trade.get('trail_stop') else '_no activado_'
    return (
        f"{'─'*30}\n🔔 *SALIDA* — {result}\n{'─'*30}\n"
        f"*{trade['symbol']}* {'🟢' if dir_=='long' else '🔴'} `{dir_.upper()}`\n"
        f"*Motivo:*   {reason_str}\n"
        f"*Entrada:*  `{entry:,.4f}`\n"
        f"*Salida:*   `{exit_price:,.4f}`\n"
        f"*SL fijo:*  `{trade['sl_fixed']:,.4f}`\n"
        f"*Trail:*    {trail_str}\n"
        f"*P/L:*      `{pnl_pct:+.3f}%` (`{pnl_usdt:+.2f} USDT`)\n"
        f"*Duración:* `{h}h {m}m`\n"
        f"{'─'*30}"
    )

async def send_tg(app: Application, text: str) -> None:
    try:
        await app.bot.send_message(
            chat_id=TELEGRAM_CHAT_ID, text=text, parse_mode='Markdown'
        )
    except Exception as e:
        log.error(f"Telegram: {e}")

# ─── COMANDOS ─────────────────────────────────────────────────────────────────
async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    global active_trade, pending_signal, pending_signal_ts
    if not active_trade:
        msg = "📭 *Sin posición abierta.*"
        if pending_signal:
            desde = (
                datetime.fromtimestamp(pending_signal_ts, tz=timezone.utc)
                .strftime('%d/%m %H:%M UTC')
                if pending_signal_ts else '?'
            )
            msg += (
                f"\n⏳ Señal pendiente: *{pending_signal.upper()}* desde {desde}"
                f"\n_Buscando primera vela 15m {'verde' if pending_signal=='long' else 'roja'}..._"
            )
    else:
        t = active_trade
        try:
            price = get_mark_price(t['symbol'])
            pnl   = (
                (price - t['entry']) / t['entry'] * 100 if t['direction'] == 'long'
                else (t['entry'] - price) / t['entry'] * 100
            )
            dur  = datetime.now(timezone.utc) - t['opened_at']
            h, m = divmod(int(dur.total_seconds()), 3600); m //= 60
            trail_str  = f"`{t['trail_stop']:,.4f}`" if t.get('trail_stop') else '_pendiente (sin swing 1h aún)_'
            msg = (
                f"📊 *Posición activa*\n\n"
                f"*{t['symbol']}* {'🟢' if t['direction']=='long' else '🔴'} `{t['direction'].upper()}`\n"
                f"Entrada:      `{t['entry']:,.4f}`\n"
                f"Precio actual:`{price:,.4f}`\n"
                f"P/L actual:   `{pnl:+.3f}%`\n"
                f"SL fijo:      `{t['sl_fixed']:,.4f}`\n"
                f"Mejor swing:  `{t['best_swing']:,.4f}`\n"
                f"Trail stop:   {trail_str}\n"
                f"Stop activo:  `{t.get('active_stop',0):,.4f}`\n"
                f"Stop Binance: `{t.get('stop_order_id','N/A')}`\n"
                f"Duración:     `{h}h {m}m`"
            )
        except Exception as e:
            msg = f"⚠️ Error status: {e}"
    await update.message.reply_text(msg, parse_mode='Markdown')

async def cmd_close(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    global active_trade
    if not active_trade:
        await update.message.reply_text("📭 Sin posición abierta.")
        return
    price = close_position_market(active_trade)
    if price:
        msg = fmt_close(active_trade, price, 'manual')
        active_trade = None
        await update.message.reply_text(msg, parse_mode='Markdown')
    else:
        await update.message.reply_text("❌ Error cerrando — verificá en Binance.")

async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    env = '🧪 TESTNET' if USE_TESTNET else '🔴 REAL'
    msg = (
        f"🤖 *Brújula Bot v2* {env}\n\n"
        f"/status — posición activa + trailing\n"
        f"/close  — cerrar manualmente\n"
        f"/help   — este mensaje\n\n"
        f"*Modelo:* EMA`{EMA_PERIOD}` · ADX`{ADX_MIN}` · 4h\n"
        f"*Entrada:* 1ª vela 15m verde/roja → close\n"
        f"*Stop:* SL`{SL_PCT}%` fijo + trail`{TRAIL_PCT}%` del swing (1h)\n"
        f"`{SYMBOL}` · `{LEVERAGE}×` · Capital`{CAPITAL_PCT}%`\n"
        f"Scan cada `{SCAN_INTERVAL}s`"
    )
    await update.message.reply_text(msg, parse_mode='Markdown')

# ─── SCAN PRINCIPAL ───────────────────────────────────────────────────────────
async def scan_job(app: Application) -> None:
    global active_trade, pending_signal, pending_signal_ts
    global last_4h_candle, last_1h_candle

    # ── GESTIÓN DE POSICIÓN ABIERTA ──────────────────────────────────────────
    if active_trade:

        # 1. ¿Binance ejecutó el stop?
        if not position_is_open(active_trade['symbol']):
            reason     = 'trailing' if active_trade.get('trail_stop') else 'sl'
            exit_price = active_trade.get('active_stop', active_trade['sl_fixed'])
            closed_trade = active_trade
            log.info(f"Posición cerrada por Binance — razón: {reason} @ {exit_price:.4f}")
            msg = fmt_close(closed_trade, exit_price, reason)
            active_trade = None
            await send_tg(app, msg)

            # ── REENTRADA: verificar si seguimos dentro de la misma vela 4h ──
            # El tester reingresa si el sesgo 4h sigue vigente y hay vela 15m confirmadora
            try:
                now_ts    = int(time.time())
                # Inicio de la vela 4h actual (múltiplo de 14400)
                vela_4h_open  = (now_ts // 14400) * 14400
                vela_4h_close = vela_4h_open + 14400

                # ¿Queda tiempo dentro de la vela 4h? (al menos 1 vela de 15m = 900s)
                tiempo_restante = vela_4h_close - now_ts
                if tiempo_restante >= 900:
                    df4h = get_klines(SYMBOL, '4h', limit=100)
                    signal, sig_ts = check_signal_4h(df4h)

                    # El sesgo debe coincidir con el trade que cerró
                    if signal == closed_trade['direction']:
                        log.info(f"Reentrada posible: sesgo {signal.upper()} sigue vigente, {tiempo_restante//60}min restantes en vela 4h")
                        # Buscar desde el timestamp de cierre del stop (exit_price moment ≈ now)
                        exit_ts = int(time.time()) - 60  # retroceder 1 min para no perder la vela actual
                        pending_signal    = signal
                        pending_signal_ts = exit_ts
                        await send_tg(app,
                            f"🔄 *Reentrada buscada* — sesgo {signal.upper()} sigue vigente\n"
                            f"Buscando 1ª vela 15m {'verde 🟢' if signal=='long' else 'roja 🔴'} "
                            f"({tiempo_restante//60}min restantes en vela 4h)"
                        )
                    else:
                        log.info(f"No reentrada: sesgo cambió o ADX insuficiente")
                        pending_signal    = None
                        pending_signal_ts = None
                else:
                    log.info(f"No reentrada: quedan solo {tiempo_restante}s en vela 4h (< 15min)")
                    pending_signal    = None
                    pending_signal_ts = None
            except Exception as e:
                log.error(f"Error evaluando reentrada: {e}")
                pending_signal    = None
                pending_signal_ts = None
            return

        # 2. Actualizar trailing si hay nueva vela de 1h cerrada
        try:
            df1h = get_klines(SYMBOL, '1h', limit=50)
            # La penúltima es la última 1h cerrada
            last_closed_1h = int(df1h['open_time'].iloc[-2]) // 1000

            if last_closed_1h != last_1h_candle:
                last_1h_candle = last_closed_1h
                log.info(f"Nueva vela 1h cerrada @ {datetime.fromtimestamp(last_closed_1h, tz=timezone.utc).strftime('%H:%M UTC')} — actualizando trailing")
                actualizado = update_trail_stop_1h(active_trade, df1h)
                if actualizado:
                    trail = active_trade['trail_stop']
                    await send_tg(app,
                        f"📈 *Trail actualizado* — {active_trade['direction'].upper()} `{SYMBOL}`\n"
                        f"Nuevo stop: `{trail:,.4f}`\n"
                        f"Swing: `{active_trade['best_swing']:,.4f}`\n"
                        f"Stop ID: `{active_trade.get('stop_order_id','N/A')}`"
                    )
            else:
                log.info(f"Trailing 1h: sin vela nueva (última: {datetime.fromtimestamp(last_closed_1h, tz=timezone.utc).strftime('%H:%M')})")
        except Exception as e:
            log.error(f"Error actualizando trailing 1h: {e}")

        log.info(
            f"Trade {active_trade['direction'].upper()} | "
            f"entry={active_trade['entry']:.4f} "
            f"stop_activo={active_trade.get('active_stop',0):.4f} "
            f"trail={'activo' if active_trade.get('trail_stop') else 'pendiente'} "
            f"stop_id={active_trade.get('stop_order_id','N/A')}"
        )
        return

    # ── SEÑAL PENDIENTE — buscar confirmación 15m ─────────────────────────────
    if pending_signal:
        try:
            df15 = get_klines(SYMBOL, '15m', limit=30)
            confirmed, entry_price, expired = find_entry_15m(
                pending_signal, pending_signal_ts, df15
            )
            if confirmed:
                # Timestamp Unix del open de la vela 15m confirmadora
                # (para filtrar velas 1h desde la entrada)
                mask = (df15['open_time'] // 1000 >= pending_signal_ts)
                post = df15[mask].iloc[:-1]
                # Encontrar la vela que dio la entrada
                entry_ts = pending_signal_ts  # fallback
                for _, row in post.iterrows():
                    cl = float(row['close']); op = float(row['open'])
                    if pending_signal == 'long'  and cl > op:
                        entry_ts = int(row['open_time']) // 1000; break
                    if pending_signal == 'short' and cl < op:
                        entry_ts = int(row['open_time']) // 1000; break

                dir_ = pending_signal
                pending_signal    = None
                pending_signal_ts = None

                trade = open_position(dir_, entry_price, entry_ts)
                if trade:
                    active_trade = trade
                    await send_tg(app, fmt_open(trade))
                else:
                    log.error("Fallo en apertura")
            elif expired:
                log.info(f"Señal {pending_signal.upper()} expirada")
                pending_signal    = None
                pending_signal_ts = None
            else:
                log.info(f"Esperando confirmación 15m para {pending_signal.upper()}")
        except Exception as e:
            log.error(f"Error buscando confirmación 15m: {e}")
        return

    # ── DETECTAR SEÑAL EN 4H (solo en vela nueva) ────────────────────────────
    try:
        df4h = get_klines(SYMBOL, '4h', limit=100)
        # Penúltima vela = última cerrada
        current_4h = int(df4h['open_time'].iloc[-2]) // 1000

        if current_4h == last_4h_candle:
            log.info("Vela 4h ya evaluada")
            return

        last_4h_candle = current_4h
        signal, sig_ts = check_signal_4h(df4h)

        if signal:
            log.info(f"Señal 4h: {signal.upper()} — ts_cierre={datetime.fromtimestamp(sig_ts, tz=timezone.utc).strftime('%d/%m %H:%M UTC')}")
            pending_signal    = signal
            pending_signal_ts = sig_ts
            await send_tg(app,
                f"🔍 *Señal 4h detectada*: `{signal.upper()}`\n"
                f"Buscando primera vela 15m {'verde 🟢' if signal=='long' else 'roja 🔴'}..."
            )
        else:
            log.info("Sin señal 4h")

    except Exception as e:
        log.error(f"Error scan 4h: {e}")

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
    log.info(f"Bot iniciado — {SYMBOL} | EMA{EMA_PERIOD} ADX{ADX_MIN} Trail{TRAIL_PCT}% | {env}")
    await app.bot.send_message(
        chat_id=TELEGRAM_CHAT_ID,
        parse_mode='Markdown',
        text=(
            f"🤖 *Brújula Bot v2* {'🧪 TESTNET' if USE_TESTNET else '🔴 REAL'}\n\n"
            f"*Señal:* `4h` — EMA`{EMA_PERIOD}` · ADX`{ADX_MIN}`\n"
            f"*Entrada:* 1ª vela `15m` verde/roja → close\n"
            f"*Stop:* `STOP_MARKET` Binance\n"
            f"  SL fijo `{SL_PCT}%` → inmediato al abrir\n"
            f"  Trail `{TRAIL_PCT}%` del swing → actualiza por cierre 1h\n"
            f"*Par:* `{SYMBOL}` · *Lev:* `{LEVERAGE}×` · *Capital:* `{CAPITAL_PCT}%`\n"
            f"*Scan:* cada `{SCAN_INTERVAL}s`\n\n"
            f"_/help para comandos_"
        )
    )

def main() -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        raise ValueError("TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID requeridos")
    if not BINANCE_KEY or not BINANCE_SECRET:
        raise ValueError("BINANCE_API_KEY y BINANCE_API_SECRET requeridos")

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
