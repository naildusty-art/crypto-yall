import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch
import io
import position_monitor as m

ACCOUNT = {'marginSummary': {'accountValue': '999.5'}, 'assetPositions': [
    {'position': {'coin': 'ETH', 'szi': '-0.0079', 'entryPx': '2527.4',
                  'unrealizedPnl': '0.45', 'liquidationPx': None}}]}

class MonitorTests(unittest.TestCase):
    def test_short_and_unavailable_prices(self):
        s = m.snapshot(ACCOUNT, {'ETH': '2470'}, 'now')
        self.assertEqual(s['positions']['ETH']['side'], 'short')
        self.assertEqual(s['positions']['ETH']['unrealized_pnl'], .45)
        self.assertIn('LINK', s['unavailable_coins'])
        self.assertTrue(s['read_only'])

    def test_only_read_endpoints(self):
        session = Mock()
        session.open.side_effect = [io.StringIO(json.dumps(ACCOUNT)), io.StringIO('{"ETH":"2470"}')]
        m.poll(session, 'public-address')
        calls = session.open.call_args_list
        self.assertEqual([json.loads(c.args[0].data)['type'] for c in calls],
                         ['clearinghouseState', 'allMids'])
        for call in calls:
            self.assertEqual(call.args[0].full_url, m.API)

    def test_errors_replace_old_snapshot(self):
        stopped = Mock(is_set=Mock(side_effect=[False, True]))
        with tempfile.TemporaryDirectory() as d, patch.object(m, 'poll', side_effect=TimeoutError), patch('builtins.print'):
            p = Path(d) / 'status.json'
            p.write_text('{"status":"healthy","positions":{"ETH":{}}}')
            m.run(Mock(), 'address', p, 30, stopped)
            s = json.loads(p.read_text())
            self.assertEqual(s['status'], 'error')
            self.assertNotIn('positions', s)
            self.assertEqual(stopped.wait.call_args.args[0] > 30, True)

    def test_reject_nonfinite_values(self):
        with self.assertRaises(ValueError):
            m.snapshot(ACCOUNT, {'ETH': 'nan'})

    def test_stop_does_not_poll(self):
        stopped = threading.Event()
        stopped.set()
        with patch.object(m, 'poll') as poll:
            m.run(Mock(), 'address', 'unused.json', 30, stopped)
            poll.assert_not_called()

if __name__ == '__main__':
    unittest.main()
