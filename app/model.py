import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort

from app.config import Settings


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Model:
    def __init__(self, settings: Settings):
        directory = settings.model_dir
        self.manifest = json.loads((directory / "manifest.json").read_text())
        self.categories = json.loads((directory / "categories.json").read_text())
        for filename in ("resnet18.onnx", "categories.json"):
            if sha256(directory / filename) != self.manifest["sha256"][filename]:
                raise ValueError(f"Artifact hash mismatch: {filename}")
        if len(self.categories) != 1000:
            raise ValueError("Expected 1000 class labels")
        options = ort.SessionOptions()
        options.intra_op_num_threads = settings.intra_op_threads
        options.inter_op_num_threads = settings.inter_op_threads
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(directory / "resnet18.onnx"),
                                            sess_options=options,
                                            providers=["CPUExecutionProvider"])
        if self.session.get_inputs()[0].shape != ["batch", 3, 224, 224]:
            raise ValueError("Unexpected model input shape")
        if self.session.get_providers() != ["CPUExecutionProvider"]:
            raise ValueError("CPUExecutionProvider is required")
        # Startup readiness includes a real warmup call, not just loading the graph.
        self.run(np.zeros((1, 3, 224, 224), dtype=np.float32))

    def run(self, batch: np.ndarray) -> tuple[np.ndarray, float]:
        started = perf_counter()
        output = self.session.run(["logits"], {"images": batch})[0]
        elapsed_ms = (perf_counter() - started) * 1000
        if output.shape != (batch.shape[0], 1000) or not np.isfinite(output).all():
            raise RuntimeError("Invalid model output")
        return output, elapsed_ms

    def predict(self, image: np.ndarray):
        logits, elapsed_ms = self.run(image[None, ...])
        shifted = logits[0] - logits[0].max()
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum()
        indices = np.argsort(-probabilities, kind="stable")[:5]
        return [{"class_id": int(i), "label": self.categories[i],
                 "probability": float(probabilities[i])} for i in indices], elapsed_ms
