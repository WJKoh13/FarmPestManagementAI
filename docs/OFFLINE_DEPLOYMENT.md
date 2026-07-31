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

Training and Ollama must not use the GPU simultaneously. Phase 1 measured only
~4.1 GB of 8 GB VRAM free with a normal desktop session, and a 3-4B model plus
a training run would contend for it. The profiles enforce the separation.

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
