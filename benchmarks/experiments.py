"""Bounded repeated experiment, rotated policy order, identical seeded arrival traces."""
import asyncio
import json
import os
from pathlib import Path
import subprocess
import time
import httpx
from benchmarks.arrivals import run, schedule
ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/'results/generated'
PAYLOAD = (ROOT/'benchmarks/fixtures/input.png').read_bytes()


def command(*args, env=None):
    return subprocess.check_output(args, cwd=ROOT, env=env, text=True, stderr=subprocess.STDOUT)


def start(mode, batch=8, delay=10, deadline=500):
    env = {**os.environ, 'INFERENCEOPS_MODE':mode, 'INFERENCEOPS_BATCH_SIZE':str(batch),
           'INFERENCEOPS_QUEUE_DELAY_MS':str(delay), 'INFERENCEOPS_REQUEST_TIMEOUT_MS':str(deadline)}
    command('docker','compose','up','-d','--no-deps','--force-recreate','api',env=env)
    with httpx.Client(trust_env=False) as client:
        for _ in range(100):
            try:
                if client.get('http://127.0.0.1:18080/health/ready').status_code == 200:
                    for _ in range(5):
                        response = client.post('http://127.0.0.1:18080/predict',content=PAYLOAD,headers={'Content-Type':'image/png'})
                        if deadline > delay:
                            response.raise_for_status()
                    return
            except httpx.HTTPError:
                pass
            time.sleep(.2)
    raise RuntimeError('API failed readiness')


def resources():
    raw = command('docker','exec','inferenceops-api-1','sh','-c','cat /sys/fs/cgroup/cpu.stat; cat /sys/fs/cgroup/memory.current')
    lines = raw.splitlines()
    return {**{k:int(v) for k,v in [line.split() for line in lines[:-1]]}, 'memory_bytes':int(lines[-1])}


def experiment(name, profile, capacity, seconds, repeat):
    before = resources()
    report = asyncio.run(run('http://127.0.0.1:18080',PAYLOAD,schedule(profile,capacity,seconds,100+repeat),seconds,DEST/(name+'.json')))
    after = resources()
    report['container_resources'] = {'before':before,'after':after,
        'cpu_cores_used':(after['usage_usec']-before['usage_usec'])/1e6/report['seconds_with_tail'],
        'throttled_seconds':(after['throttled_usec']-before['throttled_usec'])/1e6}
    report.update(profile=profile,repeat=repeat,capacity_reference_rps=capacity)
    (DEST/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(name, json.dumps(report['summary']),flush=True)
    return report


if __name__ == '__main__':
    DEST.mkdir(parents=True, exist_ok=True)
    start('immediate')
    # Sequential warm reference capacity, not an assumption from native Mac timings.
    with httpx.Client(trust_env=False) as client:
        started=time.perf_counter()
        for _ in range(50):
            client.post('http://127.0.0.1:18080/predict',content=PAYLOAD,headers={'Content-Type':'image/png'}).raise_for_status()
        capacity=50/(time.perf_counter()-started)
    (DEST/'environment.json').write_text(json.dumps({'calibration_rps':capacity,
       'architecture':command('docker','info','--format','{{.Architecture}}').strip(),
       'image_id':command('docker','inspect','--format','{{.Image}}','inferenceops-api-1').strip(),
       'api_cpu_limit':2, 'api_memory_bytes':1073741824, 'intra_op_threads':2, 'inter_op_threads':1},indent=2))
    print('calibrated reference rps',capacity,flush=True)
    modes=['immediate','fixed','dynamic']
    for repeat in range(3):
        for mode in modes[repeat:]+modes[:repeat]:
            start(mode)
            for profile in ['low','moderate','saturation','burst']:
                experiment(f'{mode}-{profile}-r{repeat}',profile,capacity,6,repeat)
    for repeat in range(3):
        for mode in ['fixed','dynamic']:
            start(mode,batch=4,delay=5)
            experiment(f'{mode}-sensitivity-r{repeat}','saturation',capacity,6,repeat)
