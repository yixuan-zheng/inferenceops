# Licensing, model provenance and attribution

## Project and dependencies

Original project code, documentation and locally generated fixtures/plots are provided under the root MIT license. This does not change third-party terms. Dependencies are installed from exact version files; containers are pulled by digest. [Dependency inventory](docs/dependency-licenses.json) records the validated package versions, roles and installed license declarations; license declarations are an inventory, not a replacement for upstream distributions' complete notices.

Key upstream components:

| Component | Upstream terms | Use |
|---|---|---|
| FastAPI | MIT | HTTP service |
| ONNX Runtime | MIT | CPU inference; notice retained in `third_party/` |
| PyTorch | BSD-style | Export/reference validation only |
| Torchvision | BSD-3-Clause | ResNet-18 model tooling and preprocessing reference; notice retained in `third_party/` |
| NumPy | BSD-3-Clause | Tensor preprocessing and analysis |
| Pillow | MIT-CMU | Image decoding/preprocessing |
| ONNX / ONNX Script | Apache-2.0 / MIT | Export tooling |
| Prometheus / Python client | Apache-2.0 | Telemetry |
| Grafana | AGPL-3.0 | Unmodified external monitoring container |
| Locust | MIT | Concurrent-user workload |
| Matplotlib | PSF-based | Performance figures |

See the upstream [Torchvision license](https://github.com/pytorch/vision/blob/main/LICENSE), [ONNX Runtime license](https://github.com/microsoft/onnxruntime/blob/main/LICENSE), [Prometheus license](https://github.com/prometheus/prometheus/blob/main/LICENSE), and [Grafana license](https://github.com/grafana/grafana/blob/main/LICENSE). Grafana is used as an unmodified separate service; its AGPL terms are not replaced by this repository's MIT license. No third-party application code is vendored beyond the required notices. Container base OS/Python packages and transitive libraries retain the notices provided within their upstream images/wheels.

## Pretrained weights and labels

The model is `torchvision.ResNet18_Weights.IMAGENET1K_V1`, sourced from:

- Checkpoint: `https://download.pytorch.org/models/resnet18-f37072fd.pth`
- SHA-256: `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`
- [Reference manifest](models/provenance/reference-manifest.json): original ONNX/label hashes, shapes, preprocessing and exporter versions.
- [Torchvision model documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).

**Neither checkpoint/ONNX binaries nor generated ImageNet class labels are redistributed here.** They are generated locally by the documented upstream download/export command. Torchvision distinguishes its code license from possible model/training-data terms in its [pretrained-model notice](https://github.com/pytorch/vision#pre-trained-model-license). Do not treat this project's MIT license as a grant to weights or ImageNet data. Evaluate upstream model/data terms for your own use, especially redistribution or commercial use. No ImageNet dataset is downloaded or included, and the performance fixture contains only locally generated noise.

ResNet architecture reference: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, *Deep Residual Learning for Image Recognition*, CVPR 2016, [paper](https://arxiv.org/abs/1512.03385). Torchvision maintainers provide the pretrained implementation/metadata. The custom scheduling/control layer and benchmark implementation are project code; normalization/resize semantics intentionally match the upstream documented transform and are validated against it.

## Images and data

The included fixture is deterministic locally generated pixel noise, with no people, private images or third-party photo licenses. Performance plots are generated from the numeric per-run public evidence. The Grafana screenshot is a real local dashboard capture containing only this application's telemetry, cropped to dashboard content (no browser profile or desktop identifiers); it is illustrative monitoring evidence, not a new performance comparison. Grafana names and marks remain their owners' property.
