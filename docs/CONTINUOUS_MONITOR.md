# Continuous testnet monitoring

Implementation is ready; deployment and an observed heartbeat are required before calling it active.

`position_monitor.py` polls Hyperliquid testnet every 30 seconds. It tracks account positions,
entry price, unrealized P&L, liquidation price, equity, and prices for the eight configured coins.
It is read-only and requires only HL_ACCOUNT_ADDRESS (public); do not provide a private key.
Account positions are not assumed to belong to this individual strategy.

The daily executor and its signals are unchanged. This monitor does not open or close orders,
enforce stops, send email, or increase trading frequency. It logs healthy/error snapshots and
atomically writes a separate status file; it never edits the executor's trading state.
Consumers must check both status and observed_at; a stopped service cannot update its heartbeat.
Allow up to two minutes before considering a healthy snapshot stale. Network errors replace
current values with an error snapshot and retry with bounded backoff (up to five minutes).

## Run on an always-on host

Use a Linux server or an always-on computer. GitHub scheduled jobs alone do not provide an
uninterrupted process. No hosting purchase or deployment is included in this change.

For a local test, use Python 3.10 or newer, export HL_ACCOUNT_ADDRESS, and run:

```sh
python position_monitor.py
```

For Linux systemd, place the checkout at /opt/crypto-yall and create a Python virtual environment at .venv (no extra dependencies are needed).
Create a dedicated crypto-monitor system user. Create /etc/crypto-yall-monitor.env containing
HL_ACCOUNT_ADDRESS=your_public_address and MONITOR_INTERVAL_SECONDS=30.
Install deploy/crypto-yall-monitor.service in /etc/systemd/system, reload systemd,
and enable/start crypto-yall-monitor. The status snapshot lives at
/var/lib/crypto-yall-monitor/status.json; logs are in the service journal.

Verify a fresh healthy timestamp, correct account positions and a restart after terminating
the process before treating deployment as complete. Keep the machine online and awake.

Offline checks: `python -m unittest test_position_monitor -v`.
