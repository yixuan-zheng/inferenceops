"""Sequential Phase 1 reference measurement; not a saturation or policy comparison."""
import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import platform
from time import perf_counter

import httpx
import numpy as np
from PIL import Image


def baseline(url: str, requests: int, warmup: int, output: Path):
    if requests < 1 or warmup < 0:
        raise ValueError("requests must be positive and warmup nonnegative")
    pixels = np.random.default_rng(42).integers(0, 256, (241, 319, 3), dtype=np.uint8)
    stream = BytesIO()
    Image.fromarray(pixels).save(stream, format="PNG")
    payload = stream.getvalue()
    output.parent.mkdir(parents=True, exist_ok=True)
    (output.parent / "baseline-input.png").write_bytes(payload)
    rows = []
    with httpx.Client(base_url=url, timeout=30, trust_env=False) as client:
        client.get("/health/ready").raise_for_status()
        info = client.get("/model/info").json()
        for i in range(warmup + requests):
            start = perf_counter()
            response = client.post("/predict", content=payload, headers={"Content-Type": "image/png"})
            elapsed = (perf_counter() - start) * 1000
            response.raise_for_status()
            data = response.json()
            if i >= warmup:
                rows.append({"request_id": data["request_id"], "client_ms": elapsed, **data["timings"]})
    summary = {}
    for key in ["client_ms", "server_total_ms", "preprocessing_ms", "preprocessing_wait_ms", "inference_wait_ms", "inference_ms"]:
        values = [row[key] for row in rows]
        summary[key] = {"p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95)),
                        "mean": float(np.mean(values))}
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
              "architecture": platform.machine(), "mode": "sequential_reference_single",
              "concurrency": 1, "warmup_requests": warmup, "measured_requests": requests,
              "successful_requests": len(rows), "failed_requests": 0,
              "model": info, "input_sha256": hashlib.sha256(payload).hexdigest(),
              "summary_ms": summary, "requests": rows,
              "limitations": "Native macOS loopback baseline; no saturation, batching, deadline, or Docker comparison."}
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("results/generated/baseline.json"))
    args = parser.parse_args()
    baseline(args.url, args.requests, args.warmup, args.output)
