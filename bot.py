"""
Brújula Trading Bot — EMA Stack
Full automático: EMA15/60/200 + ADX + ATR + RSI + Volumen en 5m
Horario: NY (13:30-20:00) + Asia (00:00-08:00) + Fin de semana
Notifica por Telegram al abrir/cerrar — sin pedir confirmación
Capitaliza ganancias entrada a entrada (usa balance real disponible)
"""

import os
import asyncio
import logging
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import requests

from telegram import Bot
from telegram.ext import Application, CommandHandler, ContextTypes, Update
from binance.client import Client
from binance.enums import *

# ─── LOGGING ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

# ─── CONFIG ──────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')
BINANCE_API_KEY  = os.environ.get('BINANCE_API_KEY', '')
BINANCE_SECRET   = os.environ.get('BINANCE_API_SECRET', '')
USE_TESTNET      = os.environ.get('USE_TESTNET', 'true').lower() == 'true'

SYMBOL        = os.environ.get('TRADING_SYMBOL', 'ETHUSDT')
LEVERAGE      = int(os.environ.get('LEVERAGE', '2'))
EMA_FAST      = int(os.environ.get('EMA_FAST',   '15'))
EMA_MID       = int(os.environ.get('EMA_MID',    '60'))
EMA_SLOW      = int(os.environ.get('EMA_SLOW',  '200'))
ADX_PERIOD    = int(os.environ.get('ADX_PERIOD', '14'))
ADX_MIN       = float(os.environ.get('ADX_MIN',  '30'))
ATR_PERIOD    = int(os.environ.get('ATR_PERIOD', '14'))
ATR_MIN_PCT   = float(os.environ.get('ATR_MIN_PCT', '0.2'))
RSI_PERIOD    = int(os.environ.get('RSI_PERIOD', '14'))
RSI_LONG_MIN  = float(os.environ.get('RSI_LONG_MIN',  '70'))
RSI_SHORT_MAX = float(os.environ.get('RSI_SHORT_MAX', '30'))
VOL_PERIOD    = int(os.environ.get('VOL_PERIOD', '50'))
VOL_MULT      = float(os.environ.get('VOL_MULT',  '1.5'))
SCAN_INTERVAL = int(os.environ.get('SCAN_INTERVAL', '60'))

# ─── STATE ───────────────────────────────────────────────────────────────────
active_trade = None

# ─── BINANCE ─────────────────────────────────────────────────────────────────
def get_client():
    c = Client(BINANCE_API_KEY, BINANCE_SECRET, testnet=USE_TESTNET)
    if USE_TESTNET:
        c.FUTURES_URL = 'https://testnet.binancefuture.com/fapi'
    return c

def fetch_klines(symbol, interval, limit):
    base = 'https://testnet.binancefuture.com' if USE_TESTNET else 'https://fapi.binance.com'
    r = requests.get(f'{base}/fapi/v1/klines',
                     params={'symbol': symbol, 'interval': interval, 'limit': limit}, timeout=15)
    r.raise_for_status()
    df = pd.DataFrame(r.json(), columns=[
        'time','open','high','low','close','vol',
        'close_time','qv','trades','taker_buy_base','taker_buy_quote','ignore'])
    for col in ['open','high','low','close','vol']:
        df[col] = df[col].astype(float)
    df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True)
    return df

def get_price(symbol):
    base = 'https://testnet.binancefuture.com' if USE_TESTNET else 'https://fapi.binance.com'
    r = requests.get(f'{base}/fapi/v1/ticker/price', params={'symbol': symbol}, timeout=10)
    r.raise_for_status()
    return float(r.json()['price'])

# ─── INDICADORES ─────────────────────────────────────────────────────────────
def calc_ema(s, p): return s.ewm(span=p, adjust=False).mean()

def calc_atr(df, p):
    hl = df['high']-df['low']
    hc = (df['high']-df['close'].shift()).abs()
    lc = (df['low']-df['close'].shift()).abs()
    return pd.concat([hl,hc,lc],axis=1).max(axis=1).ewm(span=p, adjust=False).mean()

def calc_adx(df, p):
    up = df['high'].diff(); dn = -df['low'].diff()
    pdm = up.where((up>dn)&(up>0), 0.0)
    mdm = dn.where((dn>up)&(dn>0), 0.0)
    hl = df['high']-df['low']
    hc = (df['high']-df['close'].shift()).abs()
    lc = (df['low']-df['close'].shift()).abs()
    tr = pd.concat([hl,hc,lc],axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/p, adjust=False).mean()
    pdi = 100*pdm.ewm(alpha=1/p, adjust=False).mean()/atr
    mdi = 100*mdm.ewm(alpha=1/p, adjust=False).mean()/atr
    dx  = (100*(pdi-mdi).abs()/(pdi+mdi)).replace([np.inf,-np.inf],0)
    return dx.ewm(alpha=1/p, adjust=False).mean()

def calc_rsi(s, p):
    d = s.diff()
    g = d.where(d>0, 0.0); l = (-d).where(d<0, 0.0)
    ag = g.ewm(alpha=1/p, adjust=False).mean()
    al = l.ewm(alpha=1/p, adjust=False).mean()
    return 100-(100/(1+ag/al.replace(0,np.nan)))

# ─── SESIÓN ──────────────────────────────────────────────────────────────────
def in_session(dt):
    dow = dt.weekday()
    hm  = dt.hour*60+dt.minute
    if dow >= 5: return True          # fin de semana
    if 0   <= hm < 480:  return True  # asia 00:00-08:00
    if 810 <= hm < 1200: return True  # ny 13:30-20:00
    return False

# ─── SEÑAL ───────────────────────────────────────────────────────────────────
def check_signal(df):
    min_bars = max(EMA_SLOW, VOL_PERIOD, ADX_PERIOD*3)+10
    if len(df) < min_bars: return None

    ef  = calc_ema(df['close'], EMA_FAST)
    em  = calc_ema(df['close'], EMA_MID)
    es  = calc_ema(df['close'], EMA_SLOW)
    atr = calc_atr(df, ATR_PERIOD)
    adx = calc_adx(df, ADX_PERIOD)
    rsi = calc_rsi(df['close'], RSI_PERIOD)

    i = -2
    c=df['close'].iloc[i]; ef_=ef.iloc[i]; em_=em.iloc[i]; es_=es.iloc[i]
    atr_=atr.iloc[i]; adx_=adx.iloc[i]; rsi_=rsi.iloc[i]

    if any(pd.isna(x) for x in [ef_,em_,es_,atr_,adx_,rsi_]): return None
    if adx_ < ADX_MIN: return None
    if (atr_/c*100) < ATR_MIN_PCT: return None
    avg_vol = df['vol'].iloc[i-VOL_PERIOD:i].mean()
    if df['vol'].iloc[i] < avg_vol*VOL_MULT: return None

    long_stack  = c>ef_ and ef_>em_ and em_>es_
    short_stack = c<ef_ and ef_<em_ and em_<es_

    if long_stack  and rsi_ < RSI_LONG_MIN:  return None
    if short_stack and rsi_ > RSI_SHORT_MAX: return None
    if long_stack:  return 'long'
    if short_stack: return 'short'
    return None

def check_exit(df, direction):
    em = calc_ema(df['close'], EMA_MID)
    c  = df['close'].iloc[-2]; em_ = em.iloc[-2]
    if pd.isna(em_): return False
    if direction=='long'  and c<em_: return True
    if direction=='short' and c>em_: return True
    return False

# ─── SIZING ──────────────────────────────────────────────────────────────────
def get_balance(client):
    for b in client.futures_account_balance():
        if b['asset']=='USDT': return float(b['balance'])
    return 0.0

def get_step(client, symbol):
    for s in client.futures_exchange_info()['symbols']:
        if s['symbol']==symbol:
            for f in s['filters']:
                if f['filterType']=='LOT_SIZE': return float(f['stepSize'])
    return 0.001

def calc_qty(balance, price, step):
    notional = balance*LEVERAGE*0.95
    qty = notional/price
    qty = qty-(qty%step)
    return round(qty, 8)

# ─── EJECUCIÓN ───────────────────────────────────────────────────────────────
def open_position(client, symbol, direction):
    try:
        client.futures_change_leverage(symbol=symbol, leverage=LEVERAGE)
        price   = get_price(symbol)
        balance = get_balance(client)
        step    = get_step(client, symbol)
        qty     = calc_qty(balance, price, step)
        if qty<=0: log.error("Qty=0"); return None
        side  = SIDE_BUY if direction=='long' else SIDE_SELL
        order = client.futures_create_order(symbol=symbol, side=side,
                                            type=ORDER_TYPE_MARKET, quantity=qty)
        fill  = float(order.get('avgPrice') or price)
        log.info(f"Abierto: {direction.upper()} {qty} {symbol} @ {fill:.2f}")
        return {'symbol':symbol,'direction':direction,'qty':qty,
                'entry':fill,'balance_in':balance,'opened_at':datetime.now(timezone.utc)}
    except Exception as e:
        log.error(f"Error abriendo: {e}"); return None

def close_position(client, trade):
    try:
        side = SIDE_SELL if trade['direction']=='long' else SIDE_BUY
        order = client.futures_create_order(symbol=trade['symbol'], side=side,
                                            type=ORDER_TYPE_MARKET,
                                            quantity=trade['qty'], reduceOnly=True)
        price = float(order.get('avgPrice') or get_price(trade['symbol']))
        log.info(f"Cerrado: {trade['symbol']} @ {price:.2f}")
        return price
    except Exception as e:
        log.error(f"Error cerrando: {e}"); return None

# ─── MENSAJES TELEGRAM ───────────────────────────────────────────────────────
def format_open(trade):
    env = '🧪 TESTNET' if USE_TESTNET else '🔴 REAL'
    de  = '🟢 LONG' if trade['direction']=='long' else '🔴 SHORT'
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

def format_close(trade, exit_price):
    pnl = (exit_price-trade['entry'])/trade['entry']*100
    if trade['direction']=='short': pnl=-pnl
    pnl_usd = trade['balance_in']*pnl/100*LEVERAGE
    dur = datetime.now(timezone.utc)-trade['opened_at']
    h=int(dur.total_seconds()//3600); m=int((dur.total_seconds()%3600)//60)
    re = '✅ WIN' if pnl>0 else '❌ LOSS'
    de = '🟢 LONG' if trade['direction']=='long' else '🔴 SHORT'
    return (
        f"{'─'*28}\n🔔 *POSICIÓN CERRADA*\n{'─'*28}\n"
        f"*Resultado:* {re}\n"
        f"*Par:*       `{trade['symbol']}`\n"
        f"*Dir:*       {de}\n"
        f"*Entrada:*   `{trade['entry']:,.2f}`\n"
        f"*Salida:*    `{exit_price:,.2f}`\n"
        f"*P/L:*       `{pnl:+.2f}%` (`{'+' if pnl_usd>=0 else ''}{pnl_usd:.2f} USDT`)\n"
        f"*Duración:*  `{h}h {m}m`\n"
        f"{'─'*28}"
    )

async def send_tg(app, text):
    try:
        await app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text, parse_mode='Markdown')
    except Exception as e:
        log.error(f"Telegram: {e}")

# ─── COMANDOS ────────────────────────────────────────────────────────────────
async def cmd_status(update, context):
    global active_trade
    if not active_trade:
        await update.message.reply_text("📭 Sin posiciones abiertas."); return
    try:
        price = get_price(active_trade['symbol'])
        pnl   = (price-active_trade['entry'])/active_trade['entry']*100
        if active_trade['direction']=='short': pnl=-pnl
        dur = datetime.now(timezone.utc)-active_trade['opened_at']
        h=int(dur.total_seconds()//3600); m=int((dur.total_seconds()%3600)//60)
        de = '🟢 LONG' if active_trade['direction']=='long' else '🔴 SHORT'
        msg = (f"📊 *Posición activa*\n\n"
               f"*{active_trade['symbol']}* {de}\n"
               f"Entrada: `{active_trade['entry']:,.2f}`\n"
               f"Precio: `{price:,.2f}`\n"
               f"P/L: `{pnl:+.2f}%`\n"
               f"Duración: `{h}h {m}m`")
    except Exception as e:
        msg = f"Error: {e}"
    await update.message.reply_text(msg, parse_mode='Markdown')

async def cmd_close(update, context):
    global active_trade
    if not active_trade:
        await update.message.reply_text("📭 Sin posiciones abiertas."); return
    exit_price = close_position(get_client(), active_trade)
    if exit_price:
        msg = format_close(active_trade, exit_price)+"\n_Cierre manual._"
        active_trade = None
        await update.message.reply_text(msg, parse_mode='Markdown')
    else:
        await update.message.reply_text("❌ Error — verificá en Binance.")

async def cmd_help(update, context):
    env = '🧪 TESTNET' if USE_TESTNET else '🔴 REAL'
    await update.message.reply_text(
        f"🤖 *Brújula EMA Stack Bot* {env}\n\n"
        f"*EMAs:* `{EMA_FAST}/{EMA_MID}/{EMA_SLOW}` | *ADX:* `{ADX_MIN}`\n"
        f"*ATR:* `{ATR_MIN_PCT}%` | *RSI:* `{RSI_LONG_MIN}/{RSI_SHORT_MAX}`\n"
        f"*Vol:* `{VOL_MULT}×` | *Lev:* `{LEVERAGE}×`\n"
        f"*Horario:* NY + Asia + Fin de semana\n\n"
        f"/status — posición actual y P/L\n"
        f"/close  — cerrar manualmente\n"
        f"/help   — este mensaje",
        parse_mode='Markdown'
    )

# ─── SCAN ────────────────────────────────────────────────────────────────────
async def scan_job(app):
    global active_trade
    now = datetime.now(timezone.utc)
    log.info(f"Scan {now.strftime('%Y-%m-%d %H:%M UTC')}")

    try:
        limit = max(EMA_SLOW, VOL_PERIOD, ADX_PERIOD*3)+50
        df    = fetch_klines(SYMBOL, '5m', limit)
    except Exception as e:
        log.error(f"Error fetch: {e}"); return

    client = get_client()

    if active_trade:
        try:
            if check_exit(df, active_trade['direction']):
                exit_price = close_position(client, active_trade)
                if exit_price:
                    await send_tg(app, format_close(active_trade, exit_price))
                    active_trade = None
                    log.info("Posición cerrada por señal")
                else:
                    log.error("Fallo cierre — reintento próximo scan")
        except Exception as e:
            log.error(f"Error exit check: {e}")
        return

    if not in_session(now):
        log.info(f"Fuera de horario ({now.strftime('%H:%M UTC')} dow={now.weekday()})")
        return

    try:
        signal = check_signal(df)
    except Exception as e:
        log.error(f"Error señal: {e}"); return

    if not signal:
        log.info("Sin señal"); return

    log.info(f"Señal: {signal.upper()} — ejecutando...")
    trade = open_position(client, SYMBOL, signal)
    if trade:
        active_trade = trade
        await send_tg(app, format_open(trade))
    else:
        log.error("Fallo apertura")

# ─── MAIN ────────────────────────────────────────────────────────────────────
async def scan_callback(context):
    await scan_job(context.application)

async def post_init(app):
    app.job_queue.run_repeating(callback=scan_callback, interval=SCAN_INTERVAL, first=10, name='scan')
    env = 'TESTNET 🧪' if USE_TESTNET else 'REAL 🔴'
    log.info(f"Bot iniciado — {SYMBOL} | {env} | Lev: {LEVERAGE}× | EMAs: {EMA_FAST}/{EMA_MID}/{EMA_SLOW}")
    await app.bot.send_message(
        chat_id=TELEGRAM_CHAT_ID, parse_mode='Markdown',
        text=(f"🤖 *Brújula EMA Stack Bot iniciado* {'🧪 TESTNET' if USE_TESTNET else '🔴 REAL'}\n\n"
              f"*Par:* `{SYMBOL}` | *Leverage:* `{LEVERAGE}×`\n"
              f"*EMAs:* `{EMA_FAST}/{EMA_MID}/{EMA_SLOW}` | *ADX:* `{ADX_MIN}`\n"
              f"*ATR:* `{ATR_MIN_PCT}%` | *RSI:* `{RSI_LONG_MIN}/{RSI_SHORT_MAX}` | *Vol:* `{VOL_MULT}×`\n"
              f"*Horario:* NY + Asia + Fin de semana\n\n"
              f"_Escaneando cada {SCAN_INTERVAL}s. /help para comandos._"))

def main():
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        raise ValueError("TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID son requeridos")
    if not BINANCE_API_KEY or not BINANCE_SECRET:
        raise ValueError("BINANCE_API_KEY y BINANCE_API_SECRET son requeridos")

    app = (Application.builder().token(TELEGRAM_TOKEN).post_init(post_init).build())
    app.add_handler(CommandHandler('status', cmd_status))
    app.add_handler(CommandHandler('close',  cmd_close))
    app.add_handler(CommandHandler('help',   cmd_help))
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

if __name__ == '__main__':
    main()
