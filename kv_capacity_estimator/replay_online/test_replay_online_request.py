"""Run with: python -m unittest -v test_replay_online_request.py"""
import asyncio
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest

from aiohttp import web
import replay_online_request as replay


class ReplayTests(unittest.IsolatedAsyncioTestCase):
    async def test_streaming_limit_concurrency_errors_and_payload(self):
        active = peak = received = 0
        seen = []

        async def status(request):
            return web.json_response({'model': 'served-model', 'requests': received,
                                      'finalized': False, 'metrics': {}})

        async def chat(request):
            nonlocal active, peak, received
            payload = await request.json()
            seen.append(payload)
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(.03)
                received += 1
                if payload['messages'][0]['content'] == 'fail':
                    return web.json_response({'error': 'test failure'}, status=400)
                return web.json_response({'usage': {'prompt_tokens': 7},
                                          'kv_capacity_estimation': {'page_accesses': 1}})
            finally:
                active -= 1

        app = web.Application()
        app.router.add_get('/kv-capacity-estimator/status', status)
        app.router.add_post('/v1/chat/completions', chat)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root / 'input.jsonl'
                def row(content):
                    return json.dumps({'messages': [{'role': 'user', 'content': content}],
                                       'model': 'original', 'stream': True,
                                       'stream_options': {'include_usage': True},
                                       'tools': [], 'chat_template_kwargs': {'enable_thinking': True}})
                source.write_text('\n' + row('ok') + '\n' + row('fail') + '\ninvalid\n' + row('ok2') + '\n' + row('must-not-send') + '\n')
                args = replay.parse_args(['--input', str(source), '--base-url', f'http://127.0.0.1:{port}/v1',
                                          '--concurrency', '2', '--num-requests', '4',
                                          '--summary-output', str(root / 'summary.json'),
                                          '--results-jsonl', str(root / 'results.jsonl')])
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    code = await replay.replay(args)
                self.assertEqual(code, 1)
                result = json.loads((root / 'summary.json').read_text())
                self.assertEqual((result['attempted'], result['succeeded'], result['failed'], result['prompt_tokens']), (4, 2, 2, 14))
                self.assertEqual(peak, 2)
                self.assertEqual(len(seen), 3)
                self.assertEqual(len((root / 'results.jsonl').read_text().splitlines()), 4)
                for payload in seen:
                    self.assertEqual(payload['model'], 'served-model')
                    self.assertFalse(payload['stream'])
                    self.assertNotIn('stream_options', payload)
                    self.assertTrue(payload['chat_template_kwargs']['enable_thinking'])
                    self.assertIn('tools', payload)
        finally:
            await runner.cleanup()

    def test_iterator_does_not_read_past_limit(self):
        def guarded():
            yield '\n'
            yield '{"messages": []}\n'
            raise AssertionError('read beyond requested limit')
        self.assertEqual(list(replay.records(guarded(), 1)), [(2, '{"messages": []}\n')])

    def test_unlimited_iterator_is_lazy(self):
        reads = []
        def guarded():
            for i in range(10):
                reads.append(i)
                yield str(i)
        iterator = replay.records(guarded(), None)
        next(iterator)
        self.assertEqual(reads, [0])
        self.assertEqual(len(list(iterator)), 9)

    def test_refuse_truncating_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'input'
            source.touch()
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                replay.parse_args(['--input', str(source), '--summary-output', str(source)])


if __name__ == '__main__':
    unittest.main()
