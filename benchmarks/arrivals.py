"""Open arrivals: scheduled independently of responses; every offered request is recorded."""
import asyncio
import hashlib
import json
from pathlib import Path
from time import perf_counter, process_time
import httpx
import numpy as np
from prometheus_client.parser import text_string_to_metric_families


def counters(text):
    return {s.name: s.value for family in text_string_to_metric_families(text) for s in family.samples if not s.labels}


def schedule(profile, capacity, seconds, seed):
    rng = np.random.default_rng(seed)
    times = []
    # Piecewise Poisson arrivals with fixed profile boundaries; identical per policy.
    segments = [(0, seconds, {'low': .1, 'moderate': .5, 'saturation': 1.5}.get(profile, .2)*capacity)]
    if profile == 'burst':
        segments = []
        for i in np.arange(0, seconds, 2):
            segments.extend([(i, min(i+.4, seconds), 3*capacity), (i+.4, min(i+2, seconds), .2*capacity)])
    for start, end, rate in segments:
        t = start
        while rate > 0:
            t += rng.exponential(1/rate)
            if t >= end:
                break
            times.append(float(t))
    return times


async def run(url, payload, times, seconds, output, slo_ms=100, max_inflight=128):
    rows, tasks, observations = [], set(), []
    async with httpx.AsyncClient(base_url=url, timeout=2, limits=httpx.Limits(max_connections=max_inflight+4), trust_env=False) as client:
        expected_response = await client.post('/predict', content=payload, headers={'Content-Type':'image/png'})
        expected_response.raise_for_status()
        expected = [p['class_id'] for p in expected_response.json()['predictions']]
        before = counters((await client.get('/metrics')).text)
        async def poll():
            while True:
                try:
                    observations.append(counters((await client.get('/metrics')).text).get('inferenceops_queue_depth', 0))
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(.1)
        poller = asyncio.create_task(poll())
        started, cpu_started = perf_counter(), process_time()
        async def request(index, intended):
            sent = perf_counter()
            row = {'index': index, 'scheduled_s': intended, 'sent_s': sent-started, 'lag_ms': (sent-started-intended)*1000}
            try:
                response = await client.post('/predict', content=payload, headers={'Content-Type':'image/png'})
                row.update(status=response.status_code, client_ms=(perf_counter()-sent)*1000)
                if response.status_code == 200:
                    data = response.json()
                    row.update(request_id=data['request_id'], timings=data['timings'], correct=[p['class_id'] for p in data['predictions']] == expected)
            except httpx.TimeoutException:
                row.update(status='client_timeout', client_ms=(perf_counter()-sent)*1000)
            except httpx.HTTPError as exc:
                row.update(status='transport_error', error=type(exc).__name__, client_ms=(perf_counter()-sent)*1000)
            rows.append(row)
        for index, intended in enumerate(times):
            await asyncio.sleep(max(0, started+intended-perf_counter()))
            if len(tasks) >= max_inflight or perf_counter()-started-intended > .05:
                rows.append({'index': index, 'scheduled_s': intended, 'status':'missed'})
                continue
            task = asyncio.create_task(request(index, intended))
            tasks.add(task)
            task.add_done_callback(tasks.discard)
        await asyncio.sleep(max(0, started+seconds-perf_counter()))
        if tasks:
            await asyncio.gather(*list(tasks))
        wall = perf_counter()-started
        generator_cpu = process_time()-cpu_started
        poller.cancel()
        try:
            await poller
        except asyncio.CancelledError:
            pass
        # Allow retained work from client/server deadlines to settle before accounting.
        for _ in range(100):
            info = (await client.get('/model/info')).json()
            if info.get('active_requests', 0) == 0:
                break
            await asyncio.sleep(.02)
        after = counters((await client.get('/metrics')).text)
    good = [r for r in rows if r['status'] == 200]
    sent = [r for r in rows if r['status'] != 'missed']
    def quantile(values):
        return {f'p{q}': float(np.percentile(values,q)) if values else None for q in [50,95]}
    delta = {k: after.get(k,0)-before.get(k,0) for k in after if k.endswith(('_sum','_count','_total'))}
    batches = delta.get('inferenceops_batch_size_count',0)
    summary = {'offered':len(times), 'sent':len(sent), 'missed':len(rows)-len(sent),
               'admitted':int(delta.get('inferenceops_admitted_total',0)), 'completed':len(good),
               'rejected':sum(r['status']==503 for r in rows),
               'timed_out':sum(r['status'] in [504,'client_timeout'] for r in rows),
               'other_errors':sum(r['status'] not in [200,503,504,'client_timeout','missed'] for r in rows),
               'deadline_compliant':sum(r['client_ms']<=slo_ms for r in good),
               'success_rps':len(good)/wall, 'deadline_compliant_rps':sum(r['client_ms']<=slo_ms for r in good)/wall,
               'completed_fraction_of_offered':len(good)/len(times) if times else 0,
               'deadline_fraction_of_offered':sum(r['client_ms']<=slo_ms for r in good)/len(times) if times else 0,
               'successful_client_ms':quantile([r['client_ms'] for r in good]),
               'all_sent_client_ms':quantile([r['client_ms'] for r in sent]),
               'queue_wait_ms':quantile([r['timings']['inference_wait_ms'] for r in good]),
               'accumulation_ms':quantile([r['timings']['accumulation_ms'] for r in good]),
               'mean_batch_size':delta.get('inferenceops_batch_size_sum',0)/batches if batches else 0,
               'peak_sampled_queue_depth':max(observations,default=0),
               'generator_cpu_seconds':generator_cpu, 'generator_cpu_fraction_one_core':generator_cpu/wall,
               'max_sent_lag_ms':max([r['lag_ms'] for r in sent],default=0),
               'prediction_mismatches':sum(not r['correct'] for r in good),
               'duplicate_request_ids':len(good)-len({r['request_id'] for r in good}),
               'active_after_drain':info.get('active_requests'), 'queue_after_drain':info.get('queue_depth')}
    assert len(rows) == len(times)
    assert summary['sent'] == summary['completed']+summary['rejected']+summary['timed_out']+summary['other_errors']
    report = {'seconds_scheduled':seconds, 'seconds_with_tail':wall, 'slo_ms':slo_ms,
              'input_sha256':hashlib.sha256(payload).hexdigest(), 'model':info, 'summary':summary,
              'metric_deltas':delta, 'requests':sorted(rows,key=lambda r:r['index'])}
    Path(output).write_text(json.dumps(report,indent=2)+'\n')
    return report
