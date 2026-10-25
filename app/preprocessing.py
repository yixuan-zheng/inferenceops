"""Pillow/NumPy equivalent of the weights' PIL input transform; no serving-time torch."""
from io import BytesIO
import warnings

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.config import Settings


class ImageInputError(ValueError):
    def __init__(self, detail: str, status_code: int = 422):
        super().__init__(detail)
        self.status_code = status_code


def preprocess(payload: bytes, settings: Settings) -> np.ndarray:
    if not payload:
        raise ImageInputError("Image body is empty")
    if len(payload) > settings.max_upload_bytes:
        raise ImageInputError("Image exceeds upload byte limit", 413)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(payload)) as image:
                if image.format not in {"JPEG", "PNG"}:
                    raise ImageInputError("Only JPEG and PNG images are supported", 415)
                width, height = image.size
                if (width * height > settings.max_image_pixels
                        or max(width, height) > settings.max_image_side):
                    raise ImageInputError("Decoded image dimensions exceed limit", 413)
                if getattr(image, "n_frames", 1) != 1:
                    raise ImageInputError("Animated images are not supported")
                image = image.convert("RGB")
                # Match torchvision Resize(256): resize the shorter edge, preserve aspect ratio.
                if width <= height:
                    size = (256, int(256 * height / width))
                else:
                    size = (int(256 * width / height), 256)
                if (size[0] * size[1] > settings.max_image_pixels
                        or max(size) > settings.max_image_side):
                    raise ImageInputError("Resized image dimensions exceed limit", 413)
                image = image.resize(size, Image.Resampling.BILINEAR)
                left, top = round((size[0] - 224) / 2), round((size[1] - 224) / 2)
                image = image.crop((left, top, left + 224, top + 224))
                tensor = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / np.float32(255)
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
                return np.ascontiguousarray((tensor - mean) / std)
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ImageInputError("Decoded image dimensions exceed safety limit", 413) from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        if isinstance(exc, ImageInputError):
            raise
        raise ImageInputError("Invalid or truncated image") from exc
