# Daily executor reliability update

The daily executor retains its existing five-day regime confirmation, 1% sizing
formula, configured position limit, leverage caps and asset shorting restrictions.

## Reporting and validation

- Every computed asset gets a decision or availability explanation.
- Entries validate contract quantity precision, rounded notional (at least $10),
  existing positions, and estimated available margin before submission.
  Orders below the minimum are skipped rather than enlarged beyond the sizing rule.
  The exchange can still reject an otherwise valid order because prices, fees and
  margin requirements change; its exact response is included in the report.
- Leverage configuration failures prevent entry.
- Summaries distinguish submitted orders, full fills, partial fills, errors,
  unresolved outcomes and skips. Skips include non-order decisions.
- Partial fills, errors and unresolved outcomes make the workflow fail after
  persisting state and sending its report. A green workflow is not proof of profit.
- A failed/partial close blocks new entries for that cycle. The position limit is
  rechecked against actual positions before each entry.

## Retry and recovery behavior

Connection errors and timeouts on reads have up to three attempts. Each order has
a unique client ID and its intent is saved before submission. If a submission
response is lost, the executor queries that ID. It never blindly resubmits an
uncertain order. Unresolved intents persist across runs and block further orders
until their outcome is resolved. An unknown ID alone does not prove no fill.
State read/write failures stop execution rather than losing ownership tracking.
Overlapping daily workflow executions are serialized.

## Three-day comparison

On testnet only, each daily run computes a separate three-day confirmation signal
beside the existing five-day signal. This comparison never sends orders. The report
lists differing assets; logs and state retain both signal sets and up to 90 snapshots.
This is a signal comparison, not a realistic paper P&L evaluation or evidence that
three-day confirmation performs better. Any future performance comparison must
account for fees, slippage, funding and risk before promoting a variant.

## Validation

Run `python -m unittest test_execution_safety -v` after installing `requirements.txt`.
These tests mock exchange/network calls and notifications; they do not use credentials
or submit trades. The daily workflow runs them before executing the bot.

This update does not add continuous position monitoring, exchange-hosted stop orders,
change the daily schedule, switch to mainnet, or increase trade sizing to meet minimums.
