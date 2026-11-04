#!/usr/bin/env python3
"""Bounded-memory JSONL replay against SGLang's online KV estimator."""
import argparse
import asyncio
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import time

import aiohttp

DEFAULT_INPUT = '/mnt/d/project/request_data/glm53common_drop/ksyun10084_glm53_20261030_1600-1700_sorted.evalscope.32600.jsonl'


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('must be greater than zero')
    return number


def positive_float(value):
    number = float(value)
    if not 0 < number < float('inf'):
        raise argparse.ArgumentTypeError('must be finite and greater than zero')
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path(DEFAULT_INPUT))
    parser.add_argument('--base-url', default='http://127.0.0.1:30000')
    parser.add_argument('--concurrency', type=positive_int, default=4)
    parser.add_argument('--num-requests', type=positive_int, help='Replay the first N nonblank records; default: all. Invalid/failed records count toward N.')
    parser.add_argument('--model', help='Override model; default: model reported by estimator status.')
    parser.add_argument('--api-key', default=os.environ.get('SGLANG_API_KEY'))
    parser.add_argument('--timeout', type=positive_float, default=600, help='Per HTTP request timeout in seconds (default: 600).')
    parser.add_argument('--log-interval', type=positive_int, default=100)
    parser.add_argument('--reset', action='store_true', help='Clear server cache history and metrics before replay.')
    parser.add_argument('--finalize', action='store_true', help='Finalize after replay; the server then rejects new chat requests until reset.')
    parser.add_argument('--summary-output', type=Path, default=Path('replay_online_summary.json'))
    parser.add_argument('--results-jsonl', type=Path, help='Optional per-record results, written incrementally without original prompts.')
    args = parser.parse_args(argv)
    # Opening outputs must never truncate the input or another output.
    paths = [args.input, args.summary_output]
    if args.results_jsonl:
        paths.append(args.results_jsonl)
    resolved = [p.resolve() for p in paths]
    if len(set(resolved)) != len(resolved):
        parser.error('input, summary-output and results-jsonl must have different paths')
    for i, left in enumerate(paths):
        for right in paths[i + 1:]:
            if left.exists() and right.exists() and os.path.samefile(left, right):
                parser.error('input and output files must not alias each other')
    return args


def records(source, limit):
    """Yield only one raw line at a time; never scan/count the whole file."""
    count = 0
    for line_number, raw in enumerate(source, 1):
        if not raw.strip():
            continue
        count += 1
        yield line_number, raw
        if limit is not None and count >= limit:
            return


def make_payload(raw, model):
    payload = json.loads(raw)
    if not isinstance(payload, dict) or not isinstance(payload.get('messages'), list):
        raise ValueError('expected a JSON object with a messages list')
    payload['model'] = model
    payload['stream'] = False
    payload.pop('stream_options', None)
    return payload


async def request_json(session, method, url, **kwargs):
    async with session.request(method, url, **kwargs) as response:
        if response.status >= 400:
            # Do not retain/print potentially huge error bodies containing prompts.
            detail = (await response.content.read(1024)).decode('utf-8', errors='replace')
            raise RuntimeError(f'HTTP {response.status}: {detail}')
        result = await response.json()
        if not isinstance(result, dict):
            raise ValueError('server returned a non-object JSON response')
        return result


async def replay(args):
    started = time.monotonic()
    counts = dict(attempted=0, succeeded=0, failed=0, prompt_tokens=0)
    base = args.base_url.rstrip('/')
    if base.endswith('/v1'):
        base = base[:-3]
    headers = {'Authorization': f'Bearer {args.api_key}'} if args.api_key else {}
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=args.concurrency)
    with ExitStack() as stack:
        source = stack.enter_context(args.input.open(encoding='utf-8'))
        output = stack.enter_context(args.results_jsonl.open('w', encoding='utf-8')) if args.results_jsonl else None
        async with aiohttp.ClientSession(headers=headers, timeout=timeout, connector=connector) as session:
            before = await request_json(session, 'GET', base + '/kv-capacity-estimator/status')
            if args.reset:
                before = await request_json(session, 'POST', base + '/kv-capacity-estimator/reset')
            if before.get('finalized'):
                raise RuntimeError('Estimator is finalized; use --reset to start a new run.')
            model = args.model or before['model']
            print(f"model={model} concurrency={args.concurrency} limit={args.num_requests or 'all'} server_existing_requests={before.get('requests', 0)}", flush=True)
            iterator = records(source, args.num_requests)
            read_lock = asyncio.Lock()

            async def worker():
                while True:
                    # Serialize reads in a thread, keeping disk IO off the event loop.
                    # At most one record per worker is retained, with no task backlog.
                    async with read_lock:
                        item = await asyncio.to_thread(next, iterator, None)
                    if item is None:
                        return
                    line_number, raw = item
                    counts['attempted'] += 1
                    record = {'line': line_number}
                    try:
                        payload = make_payload(raw, model)
                        del raw
                        response = await request_json(session, 'POST', base + '/v1/chat/completions', json=payload)
                        del payload
                        analysis = response.get('kv_capacity_estimation')
                        if not isinstance(analysis, dict):
                            raise ValueError('response missing kv_capacity_estimation; check estimator mode')
                        counts['succeeded'] += 1
                        counts['prompt_tokens'] += response.get('usage', {}).get('prompt_tokens', 0)
                        record.update(ok=True, usage=response.get('usage'), kv_capacity_estimation=analysis)
                    except (ValueError, RuntimeError, aiohttp.ClientError, asyncio.TimeoutError) as error:
                        counts['failed'] += 1
                        record.update(ok=False, error=str(error))
                        print(f"line {line_number}: {error}", file=sys.stderr, flush=True)
                    if output:
                        output.write(json.dumps(record, ensure_ascii=False) + '\n')
                        output.flush()
                    completed = counts['succeeded'] + counts['failed']
                    if completed % args.log_interval == 0:
                        elapsed = time.monotonic() - started
                        print(f"completed={completed} succeeded={counts['succeeded']} failed={counts['failed']} elapsed={elapsed:.1f}s rate={completed / elapsed:.2f} req/s", flush=True)
                    # Drop large objects before reading the next request.
                    # Payload can remain bound on an HTTP failure; clear it too.
                    payload = response = record = analysis = raw = None

            workers = [asyncio.create_task(worker()) for _ in range(args.concurrency)]
            try:
                await asyncio.gather(*workers)
            finally:
                for task in workers:
                    task.cancel()
                await asyncio.gather(*workers, return_exceptions=True)
            after = await request_json(session, 'GET', base + '/kv-capacity-estimator/status')
            summary = {'input': str(args.input), 'concurrency': args.concurrency,
                       'elapsed_seconds': time.monotonic() - started, **counts,
                       'server_before': before, 'server_after': after}
            if args.finalize:
                summary['finalized_result'] = await request_json(session, 'POST', base + '/kv-capacity-estimator/finalize')
            args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            print(f"Replay: {counts}; summary: {args.summary_output}", flush=True)
            print('Server cumulative hit rates (includes existing history unless --reset):')
            print(f"{'Capacity':>12} {'Page hit rate':>15} {'Prefix hit rate':>17}")
            for capacity in after.get('metrics', {}).get('configured_capacities', []):
                print(f"{capacity['capacity']:>12} {capacity['page_hit_rate']:>15.4%} {capacity['prefix_hit_rate']:>17.4%}")
    return 1 if counts['failed'] else 0


def main():
    args = parse_args()
    try:
        return asyncio.run(replay(args))
    except KeyboardInterrupt:
        print('Interrupted. In-flight requests may already have reached the server; inspect /kv-capacity-estimator/status.', file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, aiohttp.ClientError, asyncio.TimeoutError) as error:
        print(f'Replay failed: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
