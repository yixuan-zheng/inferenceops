# ResNet-18 artifact provenance

Run `python -m models.export` before Docker build or tests. The official `torchvision.ResNet18_Weights.IMAGENET1K_V1` checkpoint is downloaded using Torch's cache; `TORCH_HOME=.cache/torch` keeps it local to the checkout. Export validates its full SHA-256, uses CPU/evaluation mode, seed 42 and the Dynamo ONNX exporter (opset 18), validates the graph with ONNX checker, and writes `models/artifacts/resnet18.onnx`, labels and manifest. Dynamic batch sizes 1/2/4/8 are validated. Runtime startup verifies artifact/label hashes and performs a CPU warmup.

The public [reference manifest](provenance/reference-manifest.json) records the exact source URL, full checkpoint/ONNX/label hashes, preprocessing, tensor shapes, package versions and export settings used for the measurements. A new export records its own current manifest; toolchain/platform differences can change the graph hash even when predictions agree. Use `python -m models.validate --checkpoint .cache/torch/hub/checkpoints/resnet18-f37072fd.pth` to verify parity rather than asserting graph bytes must match on every platform.

Checkpoint and ONNX binaries are not redistributed in this repository. Labels are generated at export time from upstream metadata. Project code licensing does not relicense weights or the ImageNet training data. See [third-party notices](../THIRD_PARTY_NOTICES.md).
