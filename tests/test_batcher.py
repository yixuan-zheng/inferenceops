import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from time import perf_counter
import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from app.batcher import Batcher
from app.main import create_app
from app.preprocessing import preprocess
from tests.conftest import image_bytes


@pytest.mark.parametrize('mode', ['immediate', 'fixed', 'dynamic'])
def test_real_policy_result_mapping(settings, model, mode):
    payloads = [image_bytes(i) for i in range(6)]
    expected = [model.predict(preprocess(p, settings))[0] for p in payloads]
    with TestClient(create_app(replace(settings, mode=mode))) as client:
        with ThreadPoolExecutor(6) as pool:
            responses = list(pool.map(lambda p: client.post('/predict', content=p, headers={'Content-Type': 'image/png'}), payloads))
        for response, wanted in zip(responses, expected):
            assert response.status_code == 200
            got = response.json()['predictions']
            assert [p['class_id'] for p in got] == [p['class_id'] for p in wanted]
            np.testing.assert_allclose([p['probability'] for p in got], [p['probability'] for p in wanted], atol=1e-6)


class Fake:
    categories = [str(i) for i in range(1000)]
    def __init__(self):
        self.calls = []
    def run(self, batch):
        self.calls.append(len(batch))
        return np.tile(np.arange(1000, dtype=np.float32), (len(batch), 1)), 1
    def predict(self, image):
        self.calls.append(1)
        return [], 1


@pytest.mark.parametrize('mode', ['immediate', 'fixed', 'dynamic'])
def test_partial_flush_and_no_hang(mode):
    async def scenario():
        with ThreadPoolExecutor(1) as executor:
            model = Fake()
            scheduler = Batcher(model, executor, mode=mode, delay=.01)
            await asyncio.wait_for(scheduler.submit(np.zeros((3, 224, 224))), 1)
            assert model.calls == [1]
            await scheduler.close()
    asyncio.run(scenario())


def test_full_dynamic_dispatch_and_capacity():
    async def scenario():
        with ThreadPoolExecutor(1) as executor:
            model = Fake()
            scheduler = Batcher(model, executor, mode='dynamic', size=2, delay=10, capacity=2)
            tasks = [asyncio.create_task(scheduler.submit(np.zeros((3, 224, 224)))) for _ in range(3)]
            await asyncio.sleep(0)
            with pytest.raises(HTTPException) as caught:
                await tasks[2]
            assert caught.value.status_code == 503
            await asyncio.wait_for(asyncio.gather(*tasks[:2]), 1)
            assert model.calls == [2]
            await scheduler.close()
    asyncio.run(scenario())


def test_expiration_cancellation_and_shutdown_settle_futures():
    async def scenario():
        with ThreadPoolExecutor(1) as executor:
            model = Fake()
            scheduler = Batcher(model, executor, mode='dynamic', delay=10)
            expired = asyncio.create_task(scheduler.submit(np.zeros((3,224,224)), perf_counter()+.01))
            with pytest.raises(HTTPException) as caught:
                await asyncio.wait_for(expired, 1)
            assert caught.value.status_code == 504
            cancelled = asyncio.Event(); cancelled.set()
            with pytest.raises(HTTPException):
                await scheduler.submit(np.zeros((3,224,224)), cancelled=cancelled)
            pending = asyncio.create_task(scheduler.submit(np.zeros((3,224,224))))
            await asyncio.sleep(0)
            await scheduler.close()
            with pytest.raises(HTTPException) as caught:
                await pending
            assert caught.value.status_code == 503
            assert not model.calls
    asyncio.run(scenario())


def test_endpoint_deadline(settings):
    with TestClient(create_app(replace(settings, mode='dynamic', queue_delay_ms=100, request_timeout_ms=20))) as client:
        response = client.post('/predict', content=image_bytes(), headers={'Content-Type': 'image/png'})
        assert response.status_code == 504


def test_fixed_full_batch_waits_for_tick():
    async def scenario():
        with ThreadPoolExecutor(1) as executor:
            model = Fake()
            scheduler = Batcher(model, executor, mode='fixed', size=2, delay=.2)
            tasks = [asyncio.create_task(scheduler.submit(np.zeros((3,224,224)))) for _ in range(2)]
            await asyncio.sleep(.01)
            assert not model.calls
            await asyncio.wait_for(asyncio.gather(*tasks), 1)
            assert model.calls == [2]
            await scheduler.close()
    asyncio.run(scenario())
