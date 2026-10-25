"""Generate auditable numerical evidence using the official checkpoint and image transforms."""
import argparse
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision.models import resnet18, ResNet18_Weights

from app.config import Settings
from app.model import Model, sha256
from app.preprocessing import preprocess


def validate(checkpoint: Path, model_dir: Path, output: Path):
    settings = Settings(model_dir=model_dir)
    runtime = Model(settings)
    if sha256(checkpoint) != runtime.manifest["weights_sha256"]:
        raise ValueError("Checkpoint hash does not match manifest")
    torch.set_num_threads(2)
    reference = resnet18(weights=None).eval()
    reference.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
    tensors, transform_checks = [], []
    for seed, (width, height, channels, format) in enumerate([
        (319, 241, 3, "PNG"), (241, 319, 3, "JPEG"), (224, 224, 1, "PNG"),
        (41, 63, 4, "PNG"), (513, 256, 3, "PNG"), (256, 513, 3, "PNG"),
        (256, 256, 3, "PNG"), (301, 401, 3, "JPEG"),
    ]):
        shape = (height, width) if channels == 1 else (height, width, channels)
        array = np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)
        stream = BytesIO()
        Image.fromarray(array).save(stream, format=format)
        payload = stream.getvalue()
        actual = preprocess(payload, settings)
        with Image.open(BytesIO(payload)) as image:
            expected = transform(image.convert("RGB")).numpy()
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
        transform_checks.append({"size": [width, height], "channels": channels, "format": format,
                                 "max_abs_error": float(np.max(np.abs(actual-expected)))})
        tensors.append(actual)
    checks = []
    for n in [1, 2, 4, 8]:
        batch = np.stack(tensors[:n])
        with torch.inference_mode():
            expected = reference(torch.from_numpy(batch)).numpy()
        actual, _ = runtime.run(batch)
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)
        singles = np.concatenate([runtime.run(row[None])[0] for row in batch])
        np.testing.assert_allclose(actual, singles, rtol=1e-3, atol=1e-4)
        assert np.array_equal(actual.argmax(1), expected.argmax(1))
        checks.append({"batch_size": n, "max_abs_logit_error": float(np.max(np.abs(actual-expected))),
                       "top1_all_equal": True, "max_abs_batched_vs_single_error": float(np.max(np.abs(actual-singles)))})
    report = {"model_sha256": runtime.manifest["sha256"]["resnet18.onnx"],
              "tolerances": {"logit_rtol": 1e-3, "logit_atol": 1e-4, "preprocessing_atol": 1e-6},
              "preprocessing_checks": transform_checks, "parity_checks": checks,
              "fixture_source": "Deterministic, locally generated images; no semantic accuracy claim",
              "providers": runtime.session.get_providers(), "versions": runtime.manifest["versions"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("models/artifacts"))
    parser.add_argument("--output", type=Path, default=Path("results/generated/parity.json"))
    args = parser.parse_args()
    validate(args.checkpoint, args.model_dir, args.output)
