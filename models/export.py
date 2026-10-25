"""Export the exact pretrained model and record provenance. Run from repository root."""
import argparse
import importlib.metadata
import json
from pathlib import Path

import onnx
import torch
from torchvision.models import resnet18, ResNet18_Weights

from app.model import sha256


def export(output: Path, checkpoint: Path | None = None):
    output.mkdir(parents=True, exist_ok=True)
    weights = ResNet18_Weights.IMAGENET1K_V1
    if checkpoint:
        if sha256(checkpoint) != "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec":
            raise ValueError("Checkpoint does not match upstream ResNet-18 SHA-256")
        model = resnet18(weights=None)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    else:
        model = resnet18(weights=weights)
        checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
    if sha256(checkpoint) != "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec":
        raise ValueError("Official checkpoint SHA-256 mismatch")
    model.eval()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    sample = torch.randn(2, 3, 224, 224)
    path = output / "resnet18.onnx"
    torch.onnx.export(model, (sample,), str(path), dynamo=True, opset_version=18,
                      input_names=["images"], output_names=["logits"], external_data=False,
                      dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=8)}})
    onnx.checker.check_model(str(path))
    (output / "categories.json").write_text(json.dumps(weights.meta["categories"], indent=2) + "\n")
    manifest = {
        "schema_version": 1, "name": "resnet18", "version": "resnet18-imagenet1k-v1-onnx-v1",
        "weights": "torchvision.ResNet18_Weights.IMAGENET1K_V1", "weights_url": weights.url,
        "weights_sha256": sha256(checkpoint),
        "source": "https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html",
        "input": {"name": "images", "shape": ["batch", 3, 224, 224], "dtype": "float32",
                  "validated_batch_sizes": [1, 2, 4, 8]},
        "output": {"name": "logits", "shape": ["batch", 1000]},
        "preprocessing": {"color": "RGB", "resize_shorter_edge": 256, "interpolation": "PIL bilinear",
                          "center_crop": [224, 224], "scale": "uint8 / 255",
                          "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
                          "exif_orientation": "not applied; matches weights PIL transform"},
        "export": {"opset": 18, "dynamo": True, "seed": 42, "external_data": False},
        "versions": {p: importlib.metadata.version(p) for p in
                     ["torch", "torchvision", "onnx", "onnxscript", "onnxruntime", "pillow", "numpy"]},
        "sha256": {p: sha256(output / p) for p in ["resnet18.onnx", "categories.json"]},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/artifacts"))
    parser.add_argument("--checkpoint", type=Path, help="Reuse a cached official checkpoint")
    args = parser.parse_args()
    export(args.output, args.checkpoint)
