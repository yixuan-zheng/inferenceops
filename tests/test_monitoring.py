import asyncio
from pathlib import Path
from benchmarks.arrivals import schedule, run
from tests.conftest import image_bytes


def test_metrics_cover_success_errors_and_admission(client):
    assert client.post('/predict', content=image_bytes(), headers={'Content-Type':'image/png'}).status_code == 200
    assert client.post('/predict', content=b'bad', headers={'Content-Type':'image/png'}).status_code == 422
    text = client.get('/metrics').text
    for name in ['request_seconds', 'queue_wait_seconds', 'accumulation_seconds', 'inference_seconds', 'batch_size', 'queue_depth']:
        assert 'inferenceops_'+name in text
    assert 'inferenceops_http_requests_total{status="200"} 1.0' in text
    assert 'inferenceops_http_requests_total{status="422"} 1.0' in text
    assert 'inferenceops_admitted_total 2.0' in text
    assert 'inferenceops_queue_depth 0.0' in text


def test_arrival_trace_is_repeatable_and_independent():
    for profile in ['low', 'moderate', 'saturation', 'burst']:
        first = schedule(profile, 40, 6, 100)
        assert first == schedule(profile,40,6,100)
        assert first == sorted(first)
        assert all(0 <= t < 6 for t in first)
    assert len(schedule('saturation',40,6,100)) > len(schedule('low',40,6,100))
