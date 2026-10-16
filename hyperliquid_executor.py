"""
hyperliquid_executor.py — Live trading executor for Hyperliquid DEX.

Translates signals from signal_utils into actual trades on Hyperliquid.
Runs daily via GitHub Actions, enforces risk guardrails, and sends
execution notifications through notifier.py.

Required environment variables:
    HL_PRIVATE_KEY      – API wallet private key (trading-only, no withdraw)
    HL_ACCOUNT_ADDRESS  – Main wallet address (0x…) that owns the funds
    HL_TESTNET          – "true" to use testnet, else mainnet
    SEGREGATED_CAPITAL  – USDC allocated to bot (e.g. "10000")
    DAILY_DD_PCT        – Max daily drawdown % before auto-pause (e.g. "5")
    MAX_POSITIONS       – Max concurrent open positions (4 aggressive)
    KILL_SWITCH         – "OFF" to halt all trading, else trades enabled
    GIST_TOKEN / GIST_ID – State persistence (same as notifier)
    GMAIL_USER / GMAIL_APP_PASSWORD / NOTIFY_EMAILS – email alerts
    TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID – telegram alerts
"""

import json
import os
import sys
import math
import time
import uuid
from html import escape
import datetime as dt
from decimal import Decimal, ROUND_DOWN

import requests
from eth_account import Account

from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
from hyperliquid.utils.types import Cloid

from data_loader import fetch_data
from indicators import compute_all
from hmm_engine import causal_hmm_regimes
from strategy import generate_signals
from backtester import get_asset_profile
from signal_utils import classify_signal


# ── Config ───────────────────────────────────────────────────────────────────

# Map yfinance tickers → Hyperliquid symbols
HL_TICKER_MAP = {
    "BTC-USD": "BTC",
    "ETH-USD": "ETH",
    "SOL-USD": "SOL",
    "AVAX-USD": "AVAX",
    "LINK-USD": "LINK",
    "SUI20947-USD": "SUI",
    "XRP-USD": "XRP",
    "HBAR-USD": "HBAR",
}

ASSETS = {
    "BTC-USD": "Bitcoin (BTC)",
    "ETH-USD": "Ethereum (ETH)",
    "SOL-USD": "Solana (SOL)",
    "AVAX-USD": "Avalanche (AVAX)",
    "LINK-USD": "Chainlink (LINK)",
    "SUI20947-USD": "Sui (SUI)",
    "XRP-USD": "XRP",
    "HBAR-USD": "Hedera (HBAR)",
}

STATE_FILENAME = "trading_state.json"
POSITION_SIZE_PCT = 0.01  # 1% of segregated capital per trade
MIN_ORDER_NOTIONAL = 10.0


def read_with_retry(call, *args, **kwargs):
    """Retry transient READ failures only. Never blindly retry order submission."""
    for attempt in range(3):
        try:
            return call(*args, **kwargs)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


# ── State Persistence (GitHub Gist) ─────────────────────────────────────────

def load_trading_state() -> dict:
    gist_token = os.environ.get("GIST_TOKEN")
    gist_id = os.environ.get("GIST_ID")
    if not gist_token or not gist_id:
        raise RuntimeError("State credentials missing; refusing to trade without ownership history")
    resp = read_with_retry(requests.get,
        f"https://api.github.com/gists/{gist_id}",
        headers={"Authorization": f"token {gist_token}"},
        timeout=15,
    )
    resp.raise_for_status()
    files = resp.json().get("files", {})
    if STATE_FILENAME not in files:
        return {}
    try:
        return json.loads(files[STATE_FILENAME]["content"])
    except Exception as e:
        raise RuntimeError("Trading state is unreadable; refusing to trade") from e


def save_trading_state(state: dict):
    gist_token = os.environ.get("GIST_TOKEN")
    gist_id = os.environ.get("GIST_ID")
    if not gist_token or not gist_id:
        raise RuntimeError("Cannot persist trading state")
    resp = requests.patch(
        f"https://api.github.com/gists/{gist_id}",
        headers={"Authorization": f"token {gist_token}"},
        json={"files": {STATE_FILENAME: {"content": json.dumps(state, indent=2)}}},
        timeout=15,
    )
    resp.raise_for_status()


# ── Hyperliquid Client ──────────────────────────────────────────────────────

def get_client():
    """Return (info, exchange, account_address)."""
    priv_key = os.environ.get("HL_PRIVATE_KEY")
    account_address = os.environ.get("HL_ACCOUNT_ADDRESS")
    is_testnet = os.environ.get("HL_TESTNET", "true").lower() == "true"

    if not priv_key or not account_address:
        raise RuntimeError("HL_PRIVATE_KEY and HL_ACCOUNT_ADDRESS required")

    base_url = constants.TESTNET_API_URL if is_testnet else constants.MAINNET_API_URL
    wallet = Account.from_key(priv_key)
    info = Info(base_url, skip_ws=True)
    exchange = Exchange(wallet, base_url, account_address=account_address)

    return info, exchange, account_address


def get_account_equity(info, address: str) -> float:
    """Total account value in USDC."""
    state = read_with_retry(info.user_state, address)
    return float(state["marginSummary"]["accountValue"])


def get_open_positions(info, address: str) -> dict:
    """Return {coin: {size, entry_px, unrealized_pnl}} for open positions."""
    state = read_with_retry(info.user_state, address)
    positions = {}
    for p in state.get("assetPositions", []):
        pos = p["position"]
        size = float(pos["szi"])
        if size == 0:
            continue
        positions[pos["coin"]] = {
            "size": size,  # signed: + long, - short
            "entry_px": float(pos["entryPx"]),
            "unrealized_pnl": float(pos["unrealizedPnl"]),
        }
    return positions


def get_mid_price(info, coin: str) -> float:
    return float(info.all_mids()[coin])


def coin_is_listed(info, coin: str) -> bool:
    """Check if a coin is available for trading on the current environment."""
    return coin in info.all_mids()


# ── Signal Computation (reuse notifier logic) ───────────────────────────────

def compute_all_signals(diagnostics=None, compare=False) -> dict:
    """Return {ticker: {action_key, regime, price, bull_conf, signal}} — aggressive mode."""
    all_data = fetch_data(tickers=list(ASSETS.keys()))
    current = {}

    for ticker in ASSETS:
        try:
            raw = all_data.get(ticker)
            if raw is None or raw.empty:
                raise ValueError("No market data returned")

            df = compute_all(raw)
            regimes, bull_probs, bear_probs = causal_hmm_regimes(df)
            profile = get_asset_profile(ticker)
            regime = regimes.iloc[-1] if len(regimes) > 0 else "Unknown"
            price = float(df["Close"].iloc[-1])
            bull_conf = float(bull_probs.iloc[-1]) if len(bull_probs) > 0 else 0.0
            bear_conf = float(bear_probs.iloc[-1]) if len(bear_probs) > 0 else 0.0

            # Aggressive mode only (per call decision)
            sig = generate_signals(
                df, regimes, bull_probs=bull_probs, bear_probs=bear_probs,
                aggressive=True, bull_leverage=profile["max_bull_leverage"],
                allow_short=profile["allow_short"], atr_mult=profile["atr_mult"],
            )
            last = int(sig["Signal"].iloc[-1])
            prev = int(sig["Signal"].iloc[-2]) if len(sig) >= 2 else last
            action_key = classify_signal(last, prev, regime)

            current[ticker] = {
                "signal": last,
                "action": action_key,
                "regime": regime,
                "price": price,
                "bull_conf": bull_conf,
                "bear_conf": bear_conf,
                "leverage": float(sig["Leverage"].iloc[-1]) if "Leverage" in sig.columns else 1.0,
            }
            if compare:
                regimes3, bull3, bear3 = causal_hmm_regimes(df, confirm_days=3)
                shadow = generate_signals(
                    df, regimes3, bull_probs=bull3, bear_probs=bear3,
                    aggressive=True, bull_leverage=profile["max_bull_leverage"],
                    allow_short=profile["allow_short"], atr_mult=profile["atr_mult"],
                )
                current[ticker]["shadow_3day"] = {
                    "signal": int(shadow["Signal"].iloc[-1]),
                    "regime": str(regimes3.iloc[-1]),
                    "action": classify_signal(int(shadow["Signal"].iloc[-1]),
                                              int(shadow["Signal"].iloc[-2]), str(regimes3.iloc[-1])),
                }
        except Exception as e:
            print(f"Error computing signal for {ticker}: {e}")
            if diagnostics is not None:
                diagnostics.append({"ticker": ticker, "action": "shadow" if ticker in current else "signal",
                                    "status": "error", "error": str(e),
                                    "reason": "Shadow comparison failed" if ticker in current else "Signal computation failed",
                                    "submitted": False})
            continue

    return current


# ── Trade Decisions ─────────────────────────────────────────────────────────

def decide_trades(signals: dict, open_positions: dict, max_positions: int) -> list[dict]:
    """
    Reconcile signals vs current positions and return list of trade intents.

    Each intent: {ticker, hl_coin, action, side, reason}
    action: "open_long" | "open_short" | "close"
    """
    trades = []

    # Step 1: Determine which current positions need to be closed
    for ticker, info in signals.items():
        hl_coin = HL_TICKER_MAP[ticker]
        current_pos = open_positions.get(hl_coin)
        action_key = info["action"]

        if current_pos is None:
            continue

        is_long = current_pos["size"] > 0
        is_short = current_pos["size"] < 0

        # Close conditions
        should_close = False
        reason = ""
        if action_key == "sell_exit" and is_long:
            should_close = True
            reason = "SELL / EXIT signal"
        elif action_key == "liquidate" and (is_long or is_short):
            should_close = True
            reason = "LIQUIDATE TO CASH signal"
        elif action_key == "cover_short" and is_short:
            should_close = True
            reason = "COVER SHORT signal"
        # If signal flipped direction, close existing
        elif action_key == "buy" and is_short:
            should_close = True
            reason = "Signal flipped long while short"
        elif action_key == "enter_short" and is_long:
            should_close = True
            reason = "Signal flipped short while long"

        if should_close:
            trades.append({
                "ticker": ticker,
                "hl_coin": hl_coin,
                "action": "close",
                "side": "long" if is_long else "short",
                "reason": reason,
            })

    # Step 2: Determine which new positions to open
    # Count positions we'll have AFTER closes
    closes_by_coin = {t["hl_coin"] for t in trades if t["action"] == "close"}
    remaining_positions = {
        c: p for c, p in open_positions.items() if c not in closes_by_coin
    }
    slots_available = max(0, max_positions - len(remaining_positions))

    # Candidate opens, sorted by confidence (highest first)
    open_candidates = []
    for ticker, info in signals.items():
        hl_coin = HL_TICKER_MAP[ticker]
        action_key = info["action"]

        # Skip if we already have a position in the right direction
        existing = remaining_positions.get(hl_coin)
        if existing:
            continue

        # Open on fresh entry (buy/enter_short) OR sync when strategy
        # says we should be holding long/short but we have no position.
        if action_key in ("buy", "hold_long"):
            reason = "BUY signal" if action_key == "buy" else "Sync to hold_long (strategy already in position)"
            open_candidates.append({
                "ticker": ticker,
                "hl_coin": hl_coin,
                "action": "open_long",
                "side": "long",
                "reason": reason,
                "confidence": info["bull_conf"],
            })
        elif action_key in ("enter_short", "hold_short"):
            reason = "ENTER SHORT signal" if action_key == "enter_short" else "Sync to hold_short (strategy already in position)"
            open_candidates.append({
                "ticker": ticker,
                "hl_coin": hl_coin,
                "action": "open_short",
                "side": "short",
                "reason": reason,
                "confidence": info["bear_conf"],
            })

    open_candidates.sort(key=lambda x: x["confidence"], reverse=True)
    trades.extend(open_candidates[:slots_available])

    return trades


# ── Order Execution ─────────────────────────────────────────────────────────

def round_size(size: float, sz_decimals: int) -> float:
    """Round position size down to the coin's size decimals."""
    if sz_decimals <= 0:
        return float(int(size))
    q = Decimal("1").scaleb(-sz_decimals)
    return float(Decimal(str(size)).quantize(q, rounding=ROUND_DOWN))


def get_size_decimals(info, coin: str) -> int:
    meta = read_with_retry(info.meta)
    for universe in meta.get("universe", []):
        if universe["name"] == coin:
            return int(universe["szDecimals"])
    raise ValueError(f"No contract metadata for {coin}")


def execute_trade(info, exchange, trade: dict, capital: float, leverage: float,
                  address=None, journal=None) -> dict:
    """Validate first; persist intent before sending; reconcile uncertain responses."""
    coin = trade["hl_coin"]
    details = {**trade, "submitted": False}
    try:
        if not address or journal is None:
            raise ValueError("Account and durable order journal are required")
        mids = read_with_retry(info.all_mids)
        if coin not in mids:
            return {**details, "status": "skipped", "reason": f"{coin} unavailable on current environment"}
        mid = float(mids[coin])
        if not math.isfinite(mid) or mid <= 0:
            raise ValueError("Invalid market price")
        before = get_open_positions(info, address).get(coin, {}).get("size", 0.0)
        closing = trade["action"] == "close"
        if closing:
            if not before:
                return {**details, "status": "skipped", "reason": "Position already closed"}
            size = abs(before)
        else:
            if before:
                return {**details, "status": "skipped", "reason": "Existing position; no duplicate entry"}
            if not all(math.isfinite(x) and x > 0 for x in (capital, leverage)):
                raise ValueError("Capital/leverage must be finite and positive")
            target = capital * POSITION_SIZE_PCT * leverage
            size = round_size(target / mid, get_size_decimals(info, coin))
            details.update(target_notional=target, requested_size=size, reference_price=mid,
                           rounded_notional=size * mid)
            if size <= 0 or size * mid < MIN_ORDER_NOTIONAL:
                return {**details, "status": "skipped", "reason":
                        f"Rounded order ${size * mid:.4f} is below ${MIN_ORDER_NOTIONAL:.2f} minimum; "
                        "not increasing approved trade size"}
            account = read_with_retry(info.user_state, address)
            available = float(account.get("withdrawable", 0))
            # Hyperliquid accepts integer leverage. Use the same integer for the margin check.
            margin_leverage = max(1, int(leverage))
            needed = size * mid / margin_leverage
            details.update(available_margin=available, estimated_margin=needed,
                           margin_leverage=margin_leverage)
            if not math.isfinite(available) or available < needed:
                return {**details, "status": "skipped", "reason":
                        f"Insufficient available margin: ${available:.2f}; estimated need ${needed:.2f} plus fees"}
            response = exchange.update_leverage(margin_leverage, coin, True)
            if not isinstance(response, dict) or response.get("status") != "ok":
                return {**details, "status": "error", "error": f"Leverage update rejected: {response}"}
        cloid = Cloid.from_str("0x" + uuid.uuid4().hex)
        details.update(cloid=cloid.to_raw(), requested_size=size, reference_price=mid,
                       before_size=before)
        # If the process crashes after sending, the next run will see this pending intent.
        journal(details)
    except Exception as e:
        return {**details, "status": "error", "error": f"Preflight failed: {e}"}

    details["submitted"] = True
    try:
        response = (exchange.market_close(coin, sz=size, px=mid, cloid=cloid) if closing else
                    exchange.market_open(coin, trade["action"] == "open_long", size, px=mid, cloid=cloid))
        return _parse_response(details, response, info, coin)
    except Exception as e:
        return reconcile_order(info, address, {**details, "error": f"Submission response unavailable: {e}"})


def reconcile_order(info, address, intent):
    """Look up the unique client ID. Unknown outcomes stay blocked across runs."""
    result = {**intent, "submitted": True, "status": "unknown"}
    try:
        response = read_with_retry(info.query_order_by_cloid, address, Cloid.from_str(intent["cloid"]))
        if response.get("status") == "order":
            order = response["order"]
            status = order["status"]
            result["exchange_status"] = status
            result["oid"] = order.get("order", {}).get("oid")
            if status == "filled":
                result.update(status="filled", fill_size=intent["requested_size"], reconciled=True)
                return result
            # Terminal rejection/cancellation may still have partial fills. Only resolve
            # after checking the actual position delta; never resend the remainder here.
            if status.endswith("Rejected") or status in ("canceled", "marginCanceled", "scheduledCancel"):
                now = get_open_positions(info, address).get(intent["hl_coin"], {}).get("size", 0.0)
                delta = abs(now - intent["before_size"])
                result.update(status="partial" if delta > 0 else "error",
                              fill_size=delta, error=f"Exchange status: {status}")
                return result
            result["error"] = f"Order status {status}; no automatic resubmission"
        else:
            result["error"] = "Order not confirmed by client ID; no automatic resubmission"
    except Exception as e:
        result["error"] = f"Order verification failed: {e}; no automatic resubmission"
    return result


def _parse_response(trade: dict, resp: dict, info, coin: str) -> dict:
    """Extract fill info from Hyperliquid response."""
    result = {**trade}
    try:
        if resp.get("status") == "ok":
            statuses = resp["response"]["data"]["statuses"]
            for s in statuses:
                if "filled" in s:
                    f = s["filled"]
                    result["fill_size"] = float(f["totalSz"])
                    if result["fill_size"] <= 0:
                        raise ValueError("Exchange returned zero fill")
                    requested = float(trade.get("requested_size", result["fill_size"]))
                    result["status"] = "filled" if result["fill_size"] >= requested * (1 - 1e-9) else "partial"
                    result["fill_price"] = float(f["avgPx"])
                    result["oid"] = f.get("oid")
                    return result
                elif "error" in s:
                    result["status"] = "error"
                    result["error"] = str(s["error"])
                    return result
            result["status"] = "unknown"
            result["error"] = f"No final fill/rejection in exchange response: {resp}"
            result["raw"] = resp
        else:
            result["status"] = "error"
            result["error"] = resp.get("response", str(resp))
    except Exception as e:
        result["status"] = "unknown"
        result["error"] = f"Parse error: {e} | raw={resp}"
    return result


# ── Guardrails ──────────────────────────────────────────────────────────────

def check_kill_switch() -> bool:
    """Return True if trading should halt."""
    return os.environ.get("KILL_SWITCH", "ON").upper() == "OFF"


def check_daily_drawdown(state: dict, current_equity: float, threshold_pct: float) -> tuple[bool, dict]:
    """
    Check if today's drawdown exceeds threshold.
    Returns (should_halt, updated_state_fragment).
    """
    today = dt.date.today().isoformat()
    day_key = f"day_start_{today}"
    day_start = state.get(day_key)

    update = {}
    if day_start is None:
        update[day_key] = current_equity
        return False, update

    drawdown_pct = (current_equity - day_start) / day_start * 100 if day_start > 0 else 0

    if drawdown_pct <= -threshold_pct:
        update["halted_today"] = today
        update["halt_reason"] = f"Daily DD {drawdown_pct:.2f}% exceeded {-threshold_pct}%"
        return True, update

    return False, update


# ── Notifications ───────────────────────────────────────────────────────────

def send_execution_notifications(results: list[dict], status_summary: str):
    """Send email + telegram notifications for trade executions."""
    if not results and not status_summary:
        return

    # Email
    try:
        _send_email(results, status_summary)
    except Exception as e:
        print(f"Email send failed: {e}")

    # Telegram
    try:
        _send_telegram(results, status_summary)
    except Exception as e:
        print(f"Telegram send failed: {e}")


def execution_summary(results):
    statuses = ("filled", "partial", "error", "unknown", "skipped")
    counts = {s: sum(r.get("status") == s for r in results) for s in statuses}
    submitted = sum(bool(r.get("submitted")) for r in results)
    return f"{submitted} submitted | " + " | ".join(f"{counts[s]} {s}" for s in statuses)


def result_details(result):
    details = [str(result.get("reason", ""))]
    if result.get("error"):
        details.append("Error: " + str(result["error"]))
    for key in ("target_notional", "rounded_notional", "requested_size", "available_margin", "estimated_margin", "cloid"):
        if key in result:
            details.append(f"{key}: {result[key]}")
    return " | ".join(details)


def _send_email(results: list[dict], status_summary: str):
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    user = os.environ.get("GMAIL_USER")
    password = os.environ.get("GMAIL_APP_PASSWORD")
    recipients = os.environ.get("NOTIFY_EMAILS", "")
    if not user or not password or not recipients:
        return

    recipient_list = [e.strip() for e in recipients.split(",")]

    rows = ""
    for r in results:
        status_color = "#1f883d" if r.get("status") == "filled" else "#cf222e"
        fill_px = f"${r['fill_price']:,.4f}" if r.get("fill_price") is not None else "—"
        fill_sz = f"{r.get('fill_size', 0):.6g}" if r.get("fill_size") is not None else "—"
        rows += f"""
        <tr>
            <td style="padding:10px;border-bottom:1px solid #e1e4e8;color:#1a1a1a;">{r['ticker']}</td>
            <td style="padding:10px;border-bottom:1px solid #e1e4e8;color:#1a1a1a;">{r['action']}</td>
            <td style="padding:10px;border-bottom:1px solid #e1e4e8;color:{status_color};font-weight:bold;">{r.get('status', '?').upper()}</td>
            <td style="padding:10px;border-bottom:1px solid #e1e4e8;color:#1a1a1a;">{fill_sz}</td>
            <td style="padding:10px;border-bottom:1px solid #e1e4e8;color:#1a1a1a;">{fill_px}</td>
            <td style="padding:10px;border-bottom:1px solid #e1e4e8;color:#1a1a1a;">{escape(result_details(r))}</td>
        </tr>"""

    html = f"""
    <div style="font-family:Arial,Helvetica,sans-serif;background:#ffffff;color:#1a1a1a;padding:24px;border:1px solid #e1e4e8;border-radius:8px;max-width:760px;">
        <h2 style="color:#0969da;margin:0 0 8px 0;">Crypto Y'all Trade Execution</h2>
        <p style="color:#57606a;margin:0 0 8px 0;">{dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</p>
        <p style="color:#1a1a1a;margin:0 0 16px 0;"><strong>Status:</strong> {escape(status_summary)}</p>
        {f'<table style="width:100%;border-collapse:collapse;margin-top:16px;background:#ffffff;"><tr style="background:#f6f8fa;color:#57606a;text-transform:uppercase;font-size:0.75em;letter-spacing:0.5px;"><th style="padding:10px;text-align:left;">Asset</th><th style="padding:10px;text-align:left;">Action</th><th style="padding:10px;text-align:left;">Status</th><th style="padding:10px;text-align:left;">Size</th><th style="padding:10px;text-align:left;">Fill Price</th><th style="padding:10px;text-align:left;">Reason</th></tr>{rows}</table>' if results else '<p style="color:#1a1a1a;">No trades executed this cycle.</p>'}
    </div>
    """

    msg = MIMEMultipart("alternative")
    summary = execution_summary(results)
    msg["Subject"] = f"[Crypto Y'all] Execution: {summary}"
    msg["From"] = user
    msg["To"] = ", ".join(recipient_list)
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(user, password)
        server.send_message(msg)
    print(f"Email sent to {recipient_list}")


def _send_telegram(results: list[dict], status_summary: str):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_ids_raw = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_ids_raw:
        return

    chat_ids = [c.strip() for c in chat_ids_raw.split(",") if c.strip()]

    lines = ["*Crypto Y'all Trade Execution*", "", f"Status: {status_summary}", ""]
    for r in results:
        status = r.get("status", "?").upper()
        lines.append(f"*{r['ticker']}* — {r['action']} [{status}]")
        if r.get("status") == "filled":
            lines.append(f"  Size: {r.get('fill_size', 0):.6g} @ ${r.get('fill_price', 0):,.2f}")
        elif r.get("error"):
            lines.append(f"  Error: {r['error']}")
        lines.append(f"  Details: {result_details(r)}")
        lines.append("")

    lines.append(f"_{dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}_")
    text = "\n".join(lines)

    for chat_id in chat_ids:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            timeout=15,
        )


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print(f"Trade executor started at {dt.datetime.utcnow().isoformat()}Z")

    # Kill switch check
    if check_kill_switch():
        print("KILL_SWITCH is OFF — halting all trading")
        send_execution_notifications([], "KILL SWITCH ACTIVE — no trades executed")
        sys.exit(0)

    try:
        info, exchange, address = get_client()
    except Exception as e:
        print(f"Failed to init Hyperliquid client: {e}")
        send_execution_notifications([], f"CLIENT INIT FAILED: {e}")
        sys.exit(1)

    # Load state and check drawdown
    state = load_trading_state()
    equity = get_account_equity(info, address)
    dd_threshold = float(os.environ.get("DAILY_DD_PCT", "5"))
    halted, state_update = check_daily_drawdown(state, equity, dd_threshold)
    state.update(state_update)

    if halted:
        msg = f"Daily drawdown triggered — halting today. {state_update.get('halt_reason')}"
        print(msg)
        send_execution_notifications([], msg)
        save_trading_state(state)
        sys.exit(0)

    # Check if already halted today
    today = dt.date.today().isoformat()
    if state.get("halted_today") == today:
        print(f"Already halted today: {state.get('halt_reason')}")
        sys.exit(0)

    # Compute signals and decide trades
    diagnostics = []
    is_testnet = os.environ.get("HL_TESTNET", "true").lower() == "true"
    signals = compute_all_signals(diagnostics, compare=is_testnet)
    open_positions = get_open_positions(info, address)
    capital = float(os.environ.get("SEGREGATED_CAPITAL", "10000"))
    max_positions = int(os.environ.get("MAX_POSITIONS", "4"))

    # Filter out assets not listed on this Hyperliquid environment
    available = set(read_with_retry(info.all_mids).keys())
    for ticker in signals:
        if HL_TICKER_MAP[ticker] not in available:
            diagnostics.append({"ticker": ticker, "action": "availability", "status": "skipped",
                                "submitted": False, "reason": "Coin unavailable on current exchange environment"})
    signals = {t: s for t, s in signals.items() if HL_TICKER_MAP[t] in available}

    # Ownership tracking: only manage positions this bot opened.
    # owned_coins is the set of coin symbols this bot currently holds.
    owned_coins = set(state.get("owned_coins", []))

    # Recover any order whose final response was lost in a prior run.
    pending = list(state.get("pending_orders", []))
    unresolved = []
    for intent in pending:
        result = reconcile_order(info, address, intent)
        diagnostics.append(result)
        if result["status"] == "unknown":
            unresolved.append(intent)
        elif result.get("fill_size", 0) > 0 and intent["action"] != "close":
            owned_coins.add(intent["hl_coin"])
    state["pending_orders"] = unresolved
    state["owned_coins"] = sorted(owned_coins)
    save_trading_state(state)
    open_positions = get_open_positions(info, address)

    # Reconcile: drop owned coins that no longer have a position on the exchange
    # (e.g., another strategy or manual action closed them). This keeps state
    # consistent with the actual Hyperliquid account.
    stale_owned = owned_coins - set(open_positions.keys())
    if stale_owned:
        print(f"Dropping stale owned coins (no position on exchange): {stale_owned}")
        owned_coins -= stale_owned

    managed_positions = {c: p for c, p in open_positions.items() if c in owned_coins}

    # Uncertain orders require review; prevent a duplicate on the next daily run too.
    trades = [] if unresolved else decide_trades(signals, managed_positions, max_positions)
    print(f"Decided on {len(trades)} trade(s) (own {len(owned_coins)} position(s))")

    results = diagnostics
    intended_coins = {t["hl_coin"] for t in trades}
    for ticker, signal in signals.items():
        if HL_TICKER_MAP[ticker] not in intended_coins:
            reason = ("Unresolved order from previous run; entries blocked" if unresolved else
                      "Existing position; no additional entry" if HL_TICKER_MAP[ticker] in open_positions else
                      "Position limit or higher-confidence candidates" if signal["action"] in
                      ("buy", "hold_long", "enter_short", "hold_short") else
                      f"No entry/exit signal: {signal['action']} ({signal['regime']})")
            results.append({"ticker": ticker, "action": "decision", "status": "skipped",
                            "submitted": False, "reason": reason})

    def journal(intent):
        state.setdefault("pending_orders", []).append(intent.copy())
        save_trading_state(state)

    failed_close = False
    for trade in trades:
        sig_info = signals.get(trade["ticker"], {})
        leverage = max(1.0, min(sig_info.get("leverage", 1.0), 3.0))
        positions_now = get_open_positions(info, address)
        owned_now = {c for c in positions_now if c in owned_coins}
        blocking_reason = None
        if trade["action"] != "close":
            if failed_close:
                blocking_reason = "Earlier close did not fully fill; new entries blocked this cycle"
            elif state.get("pending_orders"):
                blocking_reason = "Unresolved order; new entries blocked"
            elif len(owned_now) >= max_positions:
                blocking_reason = "Maximum positions reached after checking actual fills"
            elif trade["hl_coin"] in positions_now:
                blocking_reason = "Existing exchange position; no duplicate entry"
        if blocking_reason:
            result = {**trade, "status": "skipped", "submitted": False, "reason": blocking_reason}
        else:
            result = execute_trade(info, exchange, trade, capital, leverage, address, journal)
        results.append(result)
        print(f"  {result['ticker']} {result['action']}: {result.get('status')} "
              f"{result.get('fill_size', '')} @ {result.get('fill_price', '')} | {result_details(result)}")

        if trade["action"] == "close" and result.get("status") != "filled":
            failed_close = True

        # Update ownership on successful fills
        if result.get("status") in ("filled", "partial"):
            coin = result["hl_coin"]
            if result["action"] == "close":
                if coin not in get_open_positions(info, address):
                    owned_coins.discard(coin)
            else:
                owned_coins.add(coin)
        if result.get("status") != "unknown":
            state["pending_orders"] = [p for p in state.get("pending_orders", [])
                                       if p.get("cloid") != result.get("cloid")]
        state["owned_coins"] = sorted(owned_coins)
        save_trading_state(state)

    # Append to trade history
    history = state.get("history", [])
    for r in results:
        history.append({
            "timestamp": dt.datetime.utcnow().isoformat() + "Z",
            **{k: v for k, v in r.items() if k not in ("raw",)},
        })
    state["history"] = history[-500:]  # keep last 500 trades
    state["last_equity"] = equity
    state["last_run"] = dt.datetime.utcnow().isoformat() + "Z"
    state["owned_coins"] = sorted(owned_coins)
    state["signal_comparison"] = {
        "timestamp": state["last_run"], "orders_use_confirm_days": 5,
        "shadow_only": True,
        "signals": {t: {"baseline": {k: s[k] for k in ("signal", "action", "regime")},
                         "shadow_3day": s.get("shadow_3day")} for t, s in signals.items()},
    } if is_testnet else {}
    comparisons = state.get("signal_comparison_history", [])
    if is_testnet:
        comparisons.append(state["signal_comparison"])
    state["signal_comparison_history"] = comparisons[-90:]
    # Show only our positions on the dashboard
    latest_positions = get_open_positions(info, address)
    state["open_positions"] = {c: p for c, p in latest_positions.items() if c in owned_coins}

    save_trading_state(state)

    summary = execution_summary(results) + f" | Equity: ${equity:,.2f}"
    if is_testnet:
        differences = [t for t, s in signals.items() if s.get("shadow_3day", {}).get("signal") != s["signal"]]
        summary += f" | Testnet 3-day shadow differs on: {', '.join(differences) or 'none'} (no shadow orders)"
        print("Signal comparison: " + json.dumps(state["signal_comparison"], default=str))
    for result in diagnostics:
        print(f"{result['ticker']}: {result['status']} | {result_details(result)}")
    send_execution_notifications(results, summary)
    print("Done")
    if any(r.get("status") in ("error", "partial", "unknown") for r in results):
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"::error::Executor failed: {e}")
        send_execution_notifications([], f"RUN FAILED: {e}")
        sys.exit(1)

