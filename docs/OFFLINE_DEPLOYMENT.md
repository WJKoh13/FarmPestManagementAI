# Offline deployment

Implemented and validated in Phase 14.

## Objective

After provisioning, the system runs with **no internet access**. No application
service may depend on an external HTTP call at request time.

## Services

| Service | Profile | Notes |
| --- | --- | --- |
| `tools` | tools | Dataset audit, manifest preparation, verification |
| `trainer` | training | Excluded from the app profile |
| `api` | app | FastAPI backend |
| `frontend` | app | Streamlit interface |
| `ollama` | app | Local LLM runtime, internal network only |

## Requirements

- Source dataset mounted **read-only**.
- Writable artifacts mounted separately from source data.
- Persistent Ollama model storage, so models survive a restart without re-pull.
- Persistent SQLite storage for knowledge and manual-evaluation data.
- Ollama exposed on the internal network only; never published to the host.
- Health checks and defined startup ordering.
- Separate app and training profiles; the trainer never starts with the app.

## GPU constraint

Training and Ollama must not use the GPU simultaneously. Free VRAM is
contended: Phase 1 measured ~4.1 GB free of 8,188 MiB under a normal desktop
session, and Phase 3 measured 7,054 MiB free when idle. A 3-4B model plus a
training run would contend for it. The profiles enforce the separation.

## GPU passthrough (verified in Phase 3)

Docker GPU access is confirmed working, so the `trainer` profile can use CUDA
inside a container:

```bash
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi
```

The container reported the RTX 4070 Laptop GPU, 8,188 MiB, driver 591.44,
compute capability 8.9. Both `--gpus all` and `--runtime=nvidia --gpus all`
work. The driver advertises CUDA 13.1, so the pinned cu126 PyTorch build is well
within range.

The image tag above is a **base** image with no CUDA toolkit or PyTorch. The
training image built in Phase 14 must install from `requirements-lock.txt`,
which carries the cu126 extra index because those wheels are not on PyPI.

## Validation checklist

Container startup, health checks, startup ordering, checkpoint loading, class
scope loading, Ollama loading, SQLite persistence, frontend-to-API networking,
API-to-Ollama networking, restart recovery, offline operation, and absence of
required external HTTP calls.

## Measurements to record

CNN latency, LLM cold-start latency, LLM warm latency, end-to-end latency, GPU
memory, CPU memory, startup time and throughput.

## Status

_Not yet implemented. Phase 14. A single documented command must start the
application._
