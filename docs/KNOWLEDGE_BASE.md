# Verified offline knowledge base

Implemented in Phase 10. SQLite, offline, with provenance for every record.

## Core principle

Treatment information is **never** generated from model memory. Every record is
ingested from an identified, approved source and carries its attribution. The
LLM summarises these records; it does not author them.

## Recorded fields

Knowledge record ID, dataset scope, original IP102 label, project label, raw
class name, canonical name, alternate names, affected crops, identification
characteristics, symptoms, monitoring, prevention, cultural controls, mechanical
controls, biological controls, organic treatment categories, escalation
conditions, safety warnings, jurisdiction, organic-certification context, source
organization, source title, source date or version, local source reference,
verification status, reviewer and review date.

## Before ingestion

Approval is required. The proposal must state the authoritative sources, the
jurisdiction they apply to, and the licensing and local-storage implications of
holding a copy offline.

## Retrieval

Deterministic lookup by class ID is preferred over similarity search: the CNN
already produces a class, and deterministic retrieval makes grounding auditable.

## Coverage is independent of CNN coverage

The CNN may support 102 classes while verified records exist for fewer. When a
predicted class has no verified record, the system:

- returns the CNN identification,
- preserves the uncertainty status,
- states that verified treatment guidance is unavailable,
- does **not** ask the LLM to invent the missing guidance,
- does **not** generate treatment claims from model memory.

The final report states the number of CNN-supported classes, the number with
verified knowledge, and which classes lack it.

## Safety constraints on stored guidance

- Organic approval claims must carry jurisdiction context.
- Dosage is stored only with an explicit verified source.
- Escalation conditions recommend expert confirmation for severe outbreaks.
- Identification content and treatment content remain separable fields, so the
  interface can show identification while withholding treatment.

## Status

_Not yet implemented. Phase 10._
