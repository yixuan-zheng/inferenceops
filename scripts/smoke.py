"""Validate the documented running deployment without private fixtures or files."""
from pathlib import Path
import json
import urllib.request
import urllib.error
import http.client
import time


def get(url, predicate=lambda data: True):
    for _ in range(60):
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                data=response.read()
            if predicate(data):
                return data
        except (urllib.error.URLError, http.client.HTTPException, OSError):
            pass
        time.sleep(.5)
    raise RuntimeError('Deployment endpoint did not become ready')


def main():
    base='http://127.0.0.1:18080'
    assert json.loads(get(base+'/health/ready'))['status']=='ready'
    info=json.loads(get(base+'/model/info'))
    assert info['execution_provider']=='CPUExecutionProvider'
    assert info['inference_workers']==1
    payload=Path('benchmarks/fixtures/input.png').read_bytes()
    request=urllib.request.Request(base+'/predict',data=payload,headers={'Content-Type':'image/png'})
    result=json.loads(urllib.request.urlopen(request,timeout=10).read())
    assert len(result['predictions'])==5
    assert result['timings']['batch_size']==1
    assert b'inferenceops_batch_size_count' in get(base+'/metrics')
    assert json.loads(get('http://127.0.0.1:13000/api/health'))['database']=='ok'
    dashboard=json.loads(get('http://127.0.0.1:13000/api/dashboards/uid/inferenceops'))
    assert dashboard['dashboard']['uid']=='inferenceops'
    up=json.loads(get('http://127.0.0.1:19090/api/v1/query?query=up', lambda data: any(row['value'][1]=='1' for row in json.loads(data)['data']['result'])))
    assert any(row['value'][1]=='1' for row in up['data']['result'])
    print('PASS: CPU inference, metrics, Prometheus scrape and provisioned Grafana dashboard')


if __name__=='__main__':
    main()
