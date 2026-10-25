from io import BytesIO

import numpy as np
from PIL import Image
import pytest
from torchvision.models import ResNet18_Weights

from app.config import Settings
from app.preprocessing import ImageInputError, preprocess
from tests.conftest import image_bytes


@pytest.mark.parametrize("size,mode,format", [
    ((319, 241), "RGB", "PNG"), ((241, 319), "RGB", "JPEG"),
    ((224, 224), "L", "PNG"), ((41, 63), "RGBA", "PNG"),
    ((513, 256), "RGB", "PNG"), ((256, 513), "RGB", "PNG"),
])
def test_matches_official_transform(settings, size, mode, format):
    payload = image_bytes(size=size, mode=mode, format=format)
    with Image.open(BytesIO(payload)) as image:
        expected = ResNet18_Weights.IMAGENET1K_V1.transforms()(image.convert("RGB")).numpy()
    actual = preprocess(payload, settings)
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=0)
    assert actual.shape == (3, 224, 224)
    assert actual.dtype == np.float32
    assert actual.flags.c_contiguous


@pytest.mark.parametrize("payload", [b"", b"not an image", b"\x89PNG\r\n\x1a\n"])
def test_invalid_images(settings, payload):
    with pytest.raises(ImageInputError):
        preprocess(payload, settings)


def test_limits_before_decode_and_resize(settings):
    with pytest.raises(ImageInputError, match="byte limit"):
        preprocess(b"x" * 11, Settings(max_upload_bytes=10))
    with pytest.raises(ImageInputError, match="dimensions") as caught:
        preprocess(image_bytes(size=(100, 100)), Settings(max_image_pixels=9999))
    assert caught.value.status_code == 413
    # A small decoded image can otherwise explode during shorter-edge resizing.
    with pytest.raises(ImageInputError, match="Resized"):
        preprocess(image_bytes(size=(8192, 1)), settings)


def test_unsupported_format(settings):
    with pytest.raises(ImageInputError) as caught:
        preprocess(image_bytes(format="BMP"), settings)
    assert caught.value.status_code == 415
