import json

import numpy as np
import pytest
import torch
from torchvision.models import resnet18

from app.config import Settings
from app.model import Model, sha256
from app.preprocessing import preprocess
from tests.conftest import image_bytes


@pytest.fixture(scope="module")
def reference(settings):
    # Export writes a local checkpoint path into neither the manifest nor the runtime.
    # Tests explicitly accept the cache location so no network is needed.
    import os
    from pathlib import Path
    checkpoint = Path(os.environ.get("INFERENCEOPS_CHECKPOINT",
                                   str(Path(torch.hub.get_dir()) / "checkpoints/resnet18-f37072fd.pth")))
    if not checkpoint.is_file():
        pytest.fail("Official checkpoint missing: export first or set INFERENCEOPS_CHECKPOINT")
    assert sha256(checkpoint) == json.loads((settings.model_dir / "manifest.json").read_text())["weights_sha256"]
    torch.set_num_threads(2)
    model = resnet18(weights=None)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model.eval()


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
def test_torch_onnx_parity(model, reference, settings, batch_size):
    batch = np.stack([preprocess(image_bytes(seed=i), settings) for i in range(batch_size)])
    with torch.inference_mode():
        expected = reference(torch.from_numpy(batch)).numpy()
    actual, _ = model.run(batch)
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_array_equal(actual.argmax(axis=1), expected.argmax(axis=1))
    singles = np.concatenate([model.run(row[None])[0] for row in batch])
    np.testing.assert_allclose(actual, singles, rtol=1e-3, atol=1e-4)


def test_hash_mismatch_fails_startup(settings, tmp_path):
    manifest = json.loads((settings.model_dir / "manifest.json").read_text())
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "categories.json").write_text((settings.model_dir / "categories.json").read_text())
    (tmp_path / "resnet18.onnx").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        Model(Settings(model_dir=tmp_path))


def test_predictions_are_finite_sorted_and_labelled(model, settings):
    predictions, elapsed = model.predict(preprocess(image_bytes(), settings))
    assert len(predictions) == 5
    assert elapsed > 0
    assert [p["probability"] for p in predictions] == sorted(
        [p["probability"] for p in predictions], reverse=True)
    for prediction in predictions:
        assert 0 <= prediction["probability"] <= 1
        assert prediction["label"] == model.categories[prediction["class_id"]]
