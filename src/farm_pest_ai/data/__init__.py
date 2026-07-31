"""Dataset manifests, audit and loading (Phases 4-5).

:mod:`~farm_pest_ai.data.manifests` translates the read-only IP102 source files
into scope-aware derived manifests. :mod:`~farm_pest_ai.data.audit` measures
integrity, duplicates, cross-split leakage and image properties.

Submodules are imported lazily so that ``audit``, which needs Pillow, does not
become a hard import requirement for code that only builds manifests.
"""

from __future__ import annotations

from .manifests import (
    SPLITS,
    ClassInfo,
    DerivedManifest,
    ManifestError,
    ManifestRecord,
    build_derived_manifest,
    manifest_csv_path,
    read_classes,
    read_derived_manifest,
    read_source_manifest,
    write_derived_manifest,
)

__all__ = [
    "SPLITS",
    "ClassInfo",
    "DerivedManifest",
    "ManifestError",
    "ManifestRecord",
    "build_derived_manifest",
    "manifest_csv_path",
    "read_classes",
    "read_derived_manifest",
    "read_source_manifest",
    "write_derived_manifest",
]
