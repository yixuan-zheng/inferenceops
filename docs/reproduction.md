# Reproduce and validate

Follow the README quickstart from a fresh checkout. It downloads the official checkpoint into `.cache/torch`, exports the model, builds a digest-pinned Linux ARM64 image from exact runtime pins and starts the provisioned monitoring stack. No model binaries, machine-specific paths, previous environments or private archives are required. Set `SSL_CERT_FILE` as documented if Python lacks a default CA bundle; TLS verification stays enabled.

## Verify the published evidence

```sh
python -m pip install -r requirements-benchmark.txt
python -m benchmarks.verify_published
INFERENCEOPS_RESULTS_DIR=results/published python -m benchmarks.analyze
```

`results/published/runs.csv` contains the 42 numeric per-run rows supporting the claims: offered/sent/admitted/outcome counts, successful and 100 ms-compliant rates, latency percentiles, batch sizes, queue/accumulation statistics, generator lag/CPU, resource measurements and repeat/profile identifiers. Summary JSON/CSV and accounting are derived. There are no usernames, absolute home paths, Docker inspection dumps or raw development logs. Original detailed traces and historical development evidence are kept in the private local archive.

Analysis aggregates run means and sample standard deviations, not request-level pseudo-confidence intervals. The analyzer supports both a numeric published CSV and fresh full generated traces. PNG/PDF regeneration writes into the selected result directory and is not needed to launch the service.

## Optional bounded rerun

This repeats the original protocol; it is not part of quickstart or CI.

```sh
python -m benchmarks.experiments
python -m benchmarks.analyze
python -m benchmarks.lifecycle
```

The experiment runner rotates policy order across three repeats (seeds 100–102), with six-second independent Poisson arrivals per profile. It calibrates 50 sequential Docker predictions, then uses 0.1×/0.5×/1.5× that rate for low/moderate/saturation. Bursts alternate 3× for 0.4 s with 0.2× until each next 2 s cycle. Identical schedules and bytes are shared across policies. Primary batch/window is eight/10 ms; fixed/dynamic saturation sensitivity is four/5 ms. One API image and fixed thread/resource limits are used for all configurations. The runner restarts only the API and warms it before measurement. Monitoring remains active.

Open arrivals are independent of responses, capped at 128 in-flight requests. Dispatches over 50 ms late are recorded as missed, not retried. Every offered request has a terminal category; sent = success + rejection + timeout + other error. Server-admitted is an independent overlapping count. Successful/deadline-compliant throughput divides by actual wall time including the completion tail; the evaluation target is 100 ms client time and the server deadline is 500 ms. Both successful and all-sent latency are saved. Queue peaks are sampled every 100 ms. Cgroup resource deltas and generator CPU/lag expose bottlenecks.

New outputs go only to ignored `results/generated/`; published summaries are preserved. A rerun uses its own measured reference capacity, so exact arrival counts and performance differ by host. Require an ARM64 host and quiet Docker resources; quotas do not isolate physical cores or thermal effects. Six-second windows and three seeds are a deliberately bounded study, not a statistically conclusive universal ranking.

Locust provides a separate concurrent-user check. It is closed-loop and its spawn rate is not a request arrival rate:

```sh
mkdir -p results/generated
locust -f benchmarks/locustfile.py --host http://127.0.0.1:18080 \
  --headless -u 12 -r 12 --run-time 8s --csv results/generated/locust
```

503s stay failures. Final JSON counters are recorded at shutdown because periodic CSV snapshots can lag. Do not compare policies using latency dominated by fast rejections, and do not run Locust/lifecycle probes concurrently with the controlled study.

CPU-only hosted CI uses an ARM64 runner, a freshly exported model and the same test/build/smoke commands. CI verifies behavior; it does not rerun the performance study. Network access to upstream packages, model weights and container registries remains required.

Linux development/export requirements use direct SHA-256-pinned official ARM64 CPU wheels for PyTorch and Torchvision, avoiding CUDA dependencies. CPython 3.11 and ARM64 are required for those wheel URLs. macOS uses the original validated native package versions.
