import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event, Lock

from fastapi.testclient import TestClient
import httpx
import numpy as np
import pytest

from app.main import create_app
from app.preprocessing import preprocess
from tests.conftest import image_bytes

HEADERS = {"Content-Type": "image/png"}


def test_health_info_and_prediction(client):
    assert client.get("/health/live").json() == {"status": "live"}
    assert client.get("/health/ready").status_code == 200
    info = client.get("/model/info").json()
    assert info["execution_provider"] == "CPUExecutionProvider"
    assert info["inference_workers"] == 1
    response = client.post("/predict", content=image_bytes(), headers=HEADERS)
    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 5
    assert body["model_version"] == info["version"]
    assert body["timings"]["server_total_ms"] >= body["timings"]["inference_ms"] > 0
    assert float(response.headers["X-Response-Time-Ms"]) > 0


def test_concurrent_requests_match_reference(client, application, settings):
    engine = application.state.engine
    session = engine.model.session
    payloads = [image_bytes(seed=i) for i in range(6)]
    expected = [engine.model.predict(preprocess(payload, settings))[0] for payload in payloads]
    with ThreadPoolExecutor(max_workers=6) as pool:
        responses = list(pool.map(lambda payload: client.post("/predict", content=payload, headers=HEADERS), payloads))
    ids = set()
    for response, wanted in zip(responses, expected):
        assert response.status_code == 200
        body = response.json()
        ids.add(body["request_id"])
        assert [p["class_id"] for p in body["predictions"]] == [p["class_id"] for p in wanted]
        np.testing.assert_allclose([p["probability"] for p in body["predictions"]],
                                   [p["probability"] for p in wanted], rtol=1e-6)
    assert len(ids) == 6
    assert engine.model.session is session
    assert engine.active == 0


@pytest.mark.parametrize("body,headers,status", [
    (b"invalid", HEADERS, 422), (b"", HEADERS, 422),
    (b"{}", {"Content-Type": "application/json"}, 415),
    (b"x", {**HEADERS, "Content-Length": "-1"}, 400),
    (b"x", {**HEADERS, "Content-Length": "nope"}, 400),
    (b"x", {**HEADERS, "Content-Length": "6000000"}, 413),
])
def test_bad_input_releases_capacity(client, application, body, headers, status):
    response = client.post("/predict", content=body, headers=headers)
    assert response.status_code == status
    assert application.state.engine.active == 0
    assert client.get("/health/ready").status_code == 200


def test_streamed_upload_limit(settings):
    application = create_app(replace(settings, max_upload_bytes=100))
    with TestClient(application) as client:
        response = client.post("/predict", content=iter([b"a" * 60, b"b" * 60]), headers=HEADERS)
        assert response.status_code == 413
        assert application.state.engine.active == 0


def test_decoded_dimensions_limit(settings):
    application = create_app(replace(settings, max_image_pixels=100))
    with TestClient(application) as client:
        assert client.post("/predict", content=image_bytes(), headers=HEADERS).status_code == 413


def test_overload_and_responsive_health(settings, monkeypatch):
    application = create_app(replace(settings, max_requests=1))
    started, release = Event(), Event()
    with TestClient(application) as client:
        original = application.state.engine.model.predict

        def blocked(tensor):
            started.set()
            assert release.wait(timeout=5)
            return original(tensor)

        monkeypatch.setattr(application.state.engine.model, "predict", blocked)
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(client.post, "/predict", content=image_bytes(), headers=HEADERS)
            try:
                assert started.wait(timeout=5)
                assert client.get("/health/live").status_code == 200
                assert client.get("/health/ready").status_code == 200
                rejected = client.post("/predict", content=image_bytes(), headers=HEADERS)
                assert rejected.status_code == 503
                assert rejected.headers["Retry-After"] == "1"
            finally:
                release.set()
            assert pending.result(timeout=5).status_code == 200
        assert application.state.engine.active == 0


def test_inference_failure_isolated(client, application, monkeypatch):
    def fail(_):
        raise RuntimeError("sensitive internal error")

    monkeypatch.setattr(application.state.engine.model, "predict", fail)
    response = client.post("/predict", content=image_bytes(), headers=HEADERS)
    assert response.status_code == 500
    assert response.json() == {"detail": "Inference failed"}
    assert application.state.engine.active == 0


def test_one_inference_and_bounded_preprocessing(client, application, monkeypatch):
    import app.main as main
    import time
    lock = Lock()
    counts = {"prep": 0, "infer": 0, "max_prep": 0, "max_infer": 0}
    original_prep = main.preprocess
    original_infer = application.state.engine.model.predict

    def wrap(kind, original):
        def run(*args):
            with lock:
                counts[kind] += 1
                counts[f"max_{kind}"] = max(counts[f"max_{kind}"], counts[kind])
            try:
                time.sleep(0.01)
                return original(*args)
            finally:
                with lock:
                    counts[kind] -= 1
        return run

    monkeypatch.setattr(main, "preprocess", wrap("prep", original_prep))
    monkeypatch.setattr(application.state.engine.model, "predict", wrap("infer", original_infer))
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(client.post, "/predict", content=image_bytes(i), headers=HEADERS) for i in range(6)]
        assert all(f.result(timeout=10).status_code == 200 for f in futures)
    assert counts["max_infer"] == 1
    assert 1 <= counts["max_prep"] <= application.state.engine.settings.preprocessing_workers


def test_cancelled_handler_retains_capacity_and_shutdown_drains(settings, monkeypatch):
    async def scenario():
        application = create_app(replace(settings, max_requests=1))
        started, release = Event(), Event()
        async with application.router.lifespan_context(application):
            engine = application.state.engine
            original = engine.model.predict

            def blocked(tensor):
                started.set()
                assert release.wait(timeout=10)
                return original(tensor)

            monkeypatch.setattr(engine.model, "predict", blocked)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url="http://test") as client:
                request = asyncio.create_task(client.post("/predict", content=image_bytes(), headers=HEADERS))
                assert await asyncio.to_thread(started.wait, 5)
                request.cancel()
                try:
                    await request
                except asyncio.CancelledError:
                    pass
                try:
                    assert engine.active == 1
                    shutdown = asyncio.create_task(engine.close())
                    await asyncio.sleep(0)
                    assert not engine.ready
                    assert not shutdown.done()
                finally:
                    release.set()
                await shutdown
                assert engine.active == 0
                assert not engine.tasks
    asyncio.run(scenario())
