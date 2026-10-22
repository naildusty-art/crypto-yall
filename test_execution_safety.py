"""Offline regression tests: never connect to an exchange or send notifications."""
import copy
import os
import unittest
from unittest.mock import Mock, patch

import requests
import numpy as np
import pandas as pd
import hyperliquid_executor as bot
import hmm_engine


TRADE = {"ticker": "BTC-USD", "hl_coin": "BTC", "action": "open_short",
         "side": "short", "reason": "test signal"}


def response(size=0.0002, price=100000):
    return {"status": "ok", "response": {"data": {"statuses": [
        {"filled": {"totalSz": str(size), "avgPx": str(price), "oid": 123}}
    ]}}}


def account(size=0):
    positions = [] if not size else [{"position": {"coin": "BTC", "szi": str(size),
                                               "entryPx": "100000", "unrealizedPnl": "0"}}]
    return {"withdrawable": "1000", "marginSummary": {"accountValue": "1000"},
            "assetPositions": positions}


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.info = Mock()
        self.info.all_mids.return_value = {"BTC": "100000"}
        self.info.meta.return_value = {"universe": [{"name": "BTC", "szDecimals": 5}]}
        self.info.user_state.return_value = account()
        self.exchange = Mock()
        self.exchange.update_leverage.return_value = {"status": "ok"}
        self.exchange.market_open.return_value = response()
        self.journal = Mock()

    def execute(self, capital=1000, leverage=2, trade=None):
        return bot.execute_trade(self.info, self.exchange, trade or TRADE,
                                 capital, leverage, "test-account", self.journal)

    def test_order_under_minimum_is_raised_to_20_dollars(self):
        self.info.all_mids.return_value = {"BTC": "83200"}
        self.exchange.market_open.return_value = response(size=0.00025, price=83200)
        result = self.execute(leverage=1)
        self.assertEqual(result["status"], "filled")
        self.assertTrue(result["raised_to_minimum"])
        self.assertEqual(result["target_notional"], 10)
        self.assertEqual(result["requested_size"], 0.00025)
        self.assertGreaterEqual(result["rounded_notional"], 20)
        self.assertAlmostEqual(result["rounded_notional"], 20.8)
        self.exchange.market_open.assert_called_once()

    def test_valid_order_is_journaled_before_submission(self):
        events = []
        self.journal.side_effect = lambda intent: events.append("journal")
        self.exchange.market_open.side_effect = lambda *a, **kw: events.append("submit") or response()
        result = self.execute()
        self.assertEqual(events, ["journal", "submit"])
        self.assertEqual(result["status"], "filled")
        self.assertEqual(result["requested_size"], 0.0002)

    def test_margin_failure_does_not_submit(self):
        self.info.user_state.return_value = {**account(), "withdrawable": "1"}
        self.assertEqual(self.execute()["status"], "skipped")
        self.exchange.market_open.assert_not_called()

    def test_leverage_failure_does_not_submit(self):
        self.exchange.update_leverage.return_value = {"status": "err", "response": "denied"}
        self.assertIn("Leverage update rejected", self.execute()["error"])
        self.exchange.market_open.assert_not_called()

    def test_existing_position_never_gets_duplicate_entry(self):
        self.info.user_state.return_value = account(-0.0002)
        self.assertEqual(self.execute()["status"], "skipped")
        self.exchange.market_open.assert_not_called()

    def test_missing_metadata_is_error(self):
        self.info.meta.return_value = {"universe": []}
        self.assertIn("No contract metadata", self.execute()["error"])
        self.exchange.market_open.assert_not_called()

    def test_nonfinite_price_never_submits(self):
        self.info.all_mids.return_value = {"BTC": "nan"}
        self.assertEqual(self.execute()["status"], "error")
        self.exchange.market_open.assert_not_called()

    def test_exact_exchange_rejection_is_preserved(self):
        self.exchange.market_open.return_value = {"status": "ok", "response": {"data": {
            "statuses": [{"error": "Insufficient margin to place order."}]}}}
        result = self.execute()
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"], "Insufficient margin to place order.")
        self.assertIn(result["error"], bot.result_details(result))
        self.assertEqual(self.exchange.market_open.call_count, 1)

    def test_partial_fill_is_not_counted_as_full(self):
        self.exchange.market_open.return_value = response(size=0.0001)
        result = self.execute()
        self.assertEqual(result["status"], "partial")
        self.assertIn("0 filled | 1 partial", bot.execution_summary([result]))

    def test_timeout_after_fill_is_reconciled_without_resubmission(self):
        self.exchange.market_open.side_effect = requests.exceptions.Timeout("lost reply")
        self.info.query_order_by_cloid.return_value = {"status": "order", "order": {
            "status": "filled", "order": {"oid": 123}}}
        result = self.execute()
        self.assertEqual(result["status"], "filled")
        self.assertTrue(result["reconciled"])
        self.assertEqual(self.exchange.market_open.call_count, 1)

    def test_unknown_timeout_is_not_retried(self):
        self.exchange.market_open.side_effect = requests.exceptions.Timeout("lost reply")
        self.info.query_order_by_cloid.return_value = {"status": "unknownOid"}
        self.assertEqual(self.execute()["status"], "unknown")
        self.assertEqual(self.exchange.market_open.call_count, 1)

    def test_malformed_reply_keeps_order_pending(self):
        self.exchange.market_open.return_value = None
        self.assertEqual(self.execute()["status"], "unknown")

    def test_cancel_after_partial_fill_is_not_retried(self):
        self.exchange.market_open.side_effect = requests.exceptions.Timeout()
        self.info.query_order_by_cloid.return_value = {"status": "order", "order": {
            "status": "canceled", "order": {"oid": 123}}}
        self.info.user_state.side_effect = [account(), account(), account(-0.0001)]
        result = self.execute()
        self.assertEqual(result["status"], "partial")
        self.assertEqual(self.exchange.market_open.call_count, 1)

    def test_failed_journal_prevents_order_submission(self):
        self.journal.side_effect = RuntimeError("state unavailable")
        self.assertEqual(self.execute()["status"], "error")
        self.exchange.market_open.assert_not_called()

    def test_close_can_be_smaller_than_entry_minimum(self):
        self.info.user_state.return_value = account(-0.00001)
        self.exchange.market_close.return_value = response(size=0.00001)
        result = self.execute(trade={**TRADE, "action": "close"})
        self.assertEqual(result["status"], "filled")
        self.exchange.market_close.assert_called_once()
        self.exchange.market_open.assert_not_called()

    def test_read_timeout_has_bounded_retries(self):
        read = Mock(side_effect=[requests.exceptions.Timeout(), {"ok": True}])
        with patch.object(bot.time, "sleep"):
            self.assertEqual(bot.read_with_retry(read), {"ok": True})
        self.assertEqual(read.call_count, 2)

    def test_position_limit_cannot_create_negative_slice(self):
        signals = {"BTC-USD": {"action": "buy", "bull_conf": 1, "bear_conf": 0}}
        self.assertEqual(bot.decide_trades(signals, {"ETH": {"size": 1}, "SOL": {"size": 1}}, 1), [])


class CycleTests(unittest.TestCase):
    def run_cycle(self, trade_result, state=None, positions=None, trades=None):
        state = copy.deepcopy(state or {})
        snapshots = []
        info, exchange = Mock(), Mock()
        info.all_mids.return_value = {"BTC": "100000", "ETH": "2500"}
        signals = {"BTC-USD": {"signal": -1, "action": "enter_short", "regime": "Bear",
                              "leverage": 2, "shadow_3day": {"signal": 1}}}
        with patch.dict(os.environ, {"HL_TESTNET": "true", "KILL_SWITCH": "ON"}), \
             patch.object(bot, "get_client", return_value=(info, exchange, "test-account")), \
             patch.object(bot, "load_trading_state", return_value=state), \
             patch.object(bot, "save_trading_state", side_effect=lambda s: snapshots.append(copy.deepcopy(s))), \
             patch.object(bot, "get_account_equity", return_value=1000), \
             patch.object(bot, "check_daily_drawdown", return_value=(False, {})), \
             patch.object(bot, "get_open_positions", return_value=positions or {}), \
             patch.object(bot, "compute_all_signals", return_value=signals) as compute, \
             patch.object(bot, "decide_trades", return_value=trades or [TRADE]), \
             patch.object(bot, "execute_trade", return_value=trade_result) as execute, \
             patch.object(bot, "reconcile_order", return_value={**TRADE, "status": "unknown", "submitted": True}), \
             patch.object(bot, "send_execution_notifications") as notify:
            exit_code = 0
            try:
                bot.main()
            except SystemExit as e:
                exit_code = e.code
        return exit_code, snapshots, execute, notify, compute

    def test_error_run_fails_after_reporting(self):
        result = {**TRADE, "status": "error", "error": "exchange rejected", "submitted": True}
        code, snapshots, execute, notify, compute = self.run_cycle(result)
        self.assertEqual(code, 1)
        self.assertIn("1 error", notify.call_args.args[1])
        self.assertIn("0 filled", notify.call_args.args[1])
        self.assertTrue(snapshots[-1]["signal_comparison"]["shadow_only"])
        self.assertTrue(compute.call_args.kwargs["compare"])
        self.assertEqual(execute.call_count, 1)  # shadow never sends an order

    def test_failed_close_blocks_new_entries(self):
        close = {**TRADE, "action": "close"}
        result = {**close, "status": "error", "error": "cannot close", "submitted": True}
        code, snapshots, execute, notify, _ = self.run_cycle(
            result, state={"owned_coins": ["BTC"]}, positions={"BTC": {"size": 1}},
            trades=[close, {**TRADE, "ticker": "ETH-USD", "hl_coin": "ETH"}])
        self.assertEqual(code, 1)
        self.assertEqual(execute.call_count, 1)
        self.assertIn("Earlier close", notify.call_args.args[0][-1]["reason"])

    def test_unresolved_order_blocks_next_run(self):
        code, snapshots, execute, notify, _ = self.run_cycle(
            {}, state={"pending_orders": [{**TRADE, "cloid": "0x" + "a" * 32}]})
        self.assertEqual(code, 1)
        execute.assert_not_called()
        self.assertTrue(snapshots[-1]["pending_orders"])


class ConfirmationTests(unittest.TestCase):
    def run_confirmation(self, days):
        frame = pd.DataFrame({"Close": range(1, 10)})
        features = pd.DataFrame({"ret": np.arange(9, dtype=float)})
        model = Mock()
        model.predict.side_effect = [np.array([s]) for s in [0, 1, 1, 1, 0, 0, 0, 0]]
        model.predict_proba.return_value = np.array([[0.5, 0.4, 0.1]])
        with patch.object(hmm_engine, "_build_features", return_value=features), \
             patch.object(hmm_engine, "GaussianHMM", return_value=model), \
             patch.object(hmm_engine, "_label_regimes_by_realized_returns", return_value={0: "Bull", 1: "Bear", 2: "Chop"}):
            return hmm_engine.causal_hmm_regimes(frame, min_train=1, confirm_days=days)[0]

    def test_three_day_variant_switches_sooner_while_default_stays_five(self):
        baseline = self.run_confirmation(5)
        shadow = self.run_confirmation(3)
        self.assertEqual(baseline.iloc[4], "Bull")
        self.assertEqual(shadow.iloc[4], "Bear")

    def test_missing_state_credentials_prevent_trading(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "State credentials missing"):
                bot.load_trading_state()


if __name__ == "__main__":
    unittest.main()
