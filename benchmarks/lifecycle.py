"""Actual Docker/HTTP lifecycle probes, run only after comparative measurements."""
import asyncio
import json
import subprocess
from time import perf_counter
import httpx
from benchmarks.experiments import start, DEST, PAYLOAD, command
from benchmarks.arrivals import counters


async def concurrent(client, count):
    async def one():
        began=perf_counter()
        try:
            r=await client.post('/predict',content=PAYLOAD,headers={'Content-Type':'image/png'})
            return {'status':r.status_code,'ms':(perf_counter()-began)*1000}
        except httpx.HTTPError as exc:
            return {'status':type(exc).__name__,'ms':(perf_counter()-began)*1000}
    return await asyncio.gather(*[one() for _ in range(count)])


async def probes():
    results={}
    await asyncio.to_thread(start,'dynamic',8,50,20)
    async with httpx.AsyncClient(base_url='http://127.0.0.1:18080',timeout=2,trust_env=False) as client:
        results['deadline_overload']=await concurrent(client,40)
        await asyncio.sleep(.5)
        results['after_deadlines']=(await client.get('/model/info')).json()
        assert results['after_deadlines']['active_requests']==0
        assert any(r['status']==504 for r in results['deadline_overload'])
        assert any(r['status']==503 for r in results['deadline_overload'])
    await asyncio.to_thread(start,'immediate')
    async with httpx.AsyncClient(base_url='http://127.0.0.1:18080',timeout=httpx.Timeout(2, read=.001),trust_env=False) as client:
        before=counters((await client.get('/metrics', timeout=2)).text)['inferenceops_admitted_total']
        results['client_abandonment']=await concurrent(client,24)
    await asyncio.sleep(1)
    async with httpx.AsyncClient(base_url='http://127.0.0.1:18080',timeout=2,trust_env=False) as client:
        results['after_abandonment']=(await client.get('/model/info')).json()
        after=counters((await client.get('/metrics')).text)['inferenceops_admitted_total']
        results['abandonment_server_admitted']=after-before
        assert after > before
        assert any(r['status']=='ReadTimeout' for r in results['client_abandonment'])
        assert results['after_abandonment']['active_requests']==0
        assert (await client.get('/health/ready')).status_code==200
        pending=asyncio.create_task(concurrent(client,40))
        await asyncio.sleep(.03)
        began=perf_counter()
        await asyncio.to_thread(command,'docker','compose','stop','api')
        results['shutdown_seconds']=perf_counter()-began
        results['shutdown_requests']=await pending
    results['shutdown_logs']=await asyncio.to_thread(command,'docker','compose','logs','--tail','20','api')
    results['notes']='All offered probe requests have recorded terminal client outcomes; shutdown transport failures are explicit. Client abandonment does not imply forced native cancellation.'
    (DEST/'lifecycle.json').write_text(json.dumps(results,indent=2)+'\n')
    assert 'Application shutdown complete' in results['shutdown_logs']
    print(json.dumps({k:v for k,v in results.items() if k.endswith('seconds')},indent=2))


if __name__=='__main__':
    asyncio.run(probes())
