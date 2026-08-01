"""Dataset manifests, audit, preprocessing and loading (Phases 4-5).

:mod:`~farm_pest_ai.data.manifests` translates the read-only IP102 source files
into scope-aware derived manifests. :mod:`~farm_pest_ai.data.audit` measures
integrity, duplicates, cross-split leakage and image properties.
:mod:`~farm_pest_ai.data.transforms` owns every pixel-level decision,
:mod:`~farm_pest_ai.data.dataset` turns a manifest into tensors, and
:mod:`~farm_pest_ai.data.loaders` assembles the ``DataLoader`` objects a
training run consumes.

Only the manifest layer is re-exported eagerly. ``audit`` needs Pillow, and
``transforms``, ``dataset`` and ``loaders`` need torch and torchvision at call
time, so importing this package stays cheap and possible in an environment
without the training extras.
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
