from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.model import Model

ROOT = Path(__file__).resolve().parents[1]


def image_bytes(seed=0, size=(319, 241), mode="RGB", format="PNG"):
    channels = {"RGB": 3, "RGBA": 4, "L": 1}[mode]
    shape = (size[1], size[0]) if channels == 1 else (size[1], size[0], channels)
    pixels = np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)
    image = Image.fromarray(pixels)
    buffer = BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()


@pytest.fixture(scope="session")
def settings():
    return Settings(model_dir=ROOT / "models/artifacts")


@pytest.fixture(scope="session")
def model(settings):
    return Model(settings)


@pytest.fixture
def application(settings):
    return create_app(settings)


@pytest.fixture
def client(application):
    with TestClient(application) as client:
        yield client
