"""Continuous, read-only Hyperliquid TESTNET monitoring. No trading credentials needed."""
import datetime as dt
import json
import math
import os
from pathlib import Path
import re
import signal
import threading
import time

import urllib.request
import urllib.error

API = 'https://api.hyperliquid-testnet.xyz/info'
COINS = ('BTC', 'ETH', 'SOL', 'AVAX', 'LINK', 'SUI', 'XRP', 'HBAR')


def number(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('Non-finite exchange value')
    return result


def snapshot(account, mids, now=None):
    positions = {}
    for item in account.get('assetPositions', []):
        p = item['position']
        size = number(p['szi'])
        if size:
            positions[p['coin']] = {
                'side': 'long' if size > 0 else 'short',
                'size': size,
                'entry_price': number(p['entryPx']),
                'unrealized_pnl': number(p['unrealizedPnl']),
                'liquidation_price': None if p.get('liquidationPx') is None else number(p['liquidationPx']),
            }
    return {
        'observed_at': now or dt.datetime.now(dt.timezone.utc).isoformat(),
        'environment': 'testnet', 'read_only': True, 'status': 'healthy',
        'equity': number(account['marginSummary']['accountValue']),
        'positions': positions,
        'prices': {c: number(mids[c]) for c in COINS if c in mids},
        'unavailable_coins': [c for c in COINS if c not in mids],
        'scope': 'Account positions; ownership by an individual strategy is not inferred',
    }


def poll(session, address):
    def read(payload):
        request = urllib.request.Request(API, data=json.dumps(payload).encode(),
                                         headers={'Content-Type': 'application/json'})
        with session.open(request, timeout=15) as response:
            return json.load(response)
    account = read({'type': 'clearinghouseState', 'user': address})
    mids = read({'type': 'allMids'})
    return snapshot(account, mids)


def save_status(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def run(session, address, path, interval, stopped):
    failures = 0
    while not stopped.is_set():
        started = time.monotonic()
        try:
            data = poll(session, address)
            failures = 0
        except (OSError, ValueError, KeyError, TypeError) as error:
            failures += 1
            # Never label old position values as current after a failed request.
            data = {'status': 'error', 'environment': 'testnet', 'read_only': True,
                    'observed_at': dt.datetime.now(dt.timezone.utc).isoformat(),
                    'consecutive_failures': failures, 'error_type': type(error).__name__}
        save_status(path, data)
        print(json.dumps(data), flush=True)
        delay = min(300, interval * 2 ** min(failures, 4)) if failures else interval
        stopped.wait(max(0, delay - (time.monotonic() - started)))


def main():
    address = os.environ.get('HL_ACCOUNT_ADDRESS', '').strip()
    if not re.fullmatch(r'0x[0-9a-fA-F]{40}', address):
        raise SystemExit('Set HL_ACCOUNT_ADDRESS to the public account address')
    if os.environ.get('HL_TESTNET', 'true').lower() != 'true':
        raise SystemExit('This monitor is testnet-only')
    interval = number(os.environ.get('MONITOR_INTERVAL_SECONDS', '30'))
    if not 10 <= interval <= 3600:
        raise SystemExit('Monitor interval must be 10–3600 seconds')
    stopped = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopped.set())
    session = urllib.request.build_opener()
    run(session, address, os.environ.get('MONITOR_STATUS_PATH', 'monitor_status.json'), interval, stopped)


if __name__ == '__main__':
    main()
