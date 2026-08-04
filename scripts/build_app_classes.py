"""Generate the app's committed class-name tables for the classification scopes.

The app needs to put a *name* on a prediction. The training side never needed
one: ``farm_pest_ai`` works in project labels and derives everything it needs
from :mod:`farm_pest_ai.scopes`. The names live in ``classes.txt`` inside the
IP102 archive, which is deliberately not committed and is absent from a bare
checkout -- so an app that read them at start-up would have no names at all on
exactly the machine the offline demo is meant to run on.

This script therefore renders them once, into ``data_manifests/classes_<scope>.json``,
in the same shape as the existing ``classes_top15.json``. Those files are
committed; the dataset is not.

**Slugs are the join key to the treatment guides.** Where a scope contains an
IP102 label that ``classes_top15.json`` already names, that file's slug is
reused verbatim rather than re-derived from ``classes.txt``. Re-deriving would
produce ``cicadellidae`` for one and ``rice_leafhopper`` for the other from the
same underlying pest and quietly break the lookup in
:mod:`app.treatment_guides`.

Run it from the repository root with the dataset present:

    python scripts/build_app_classes.py
    python scripts/build_app_classes.py --check

``--check`` re-renders and compares bytes without writing, so CI can prove the
committed files still match their source.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from farm_pest_ai.scopes import (  # noqa: E402
    CLASS_MAPPING_VERSION,
    FULL102,
    RICE10,
    ScopeSpec,
)

#: Scopes rendered by this script. The detection scopes are excluded: their
#: labels come from ``splits_top*.json``, not from ``classes.txt``, and
#: ``classes_top15.json`` already covers the one the app serves.
SCOPES: tuple[ScopeSpec, ...] = (RICE10, FULL102)

DEFAULT_CLASSES_FILE = PROJECT_ROOT / "ip102_v1.1" / "Classification" / "classes.txt"
MANIFEST_DIR = PROJECT_ROOT / "data_manifests"
GUIDED_CLASSES_FILE = MANIFEST_DIR / "classes_top15.json"

DESCRIPTION = (
    "Class names for scope {scope!r}, rendered by scripts/build_app_classes.py from "
    "ip102_v1.1/Classification/classes.txt. class_name is the slug used as the "
    "treatment-guide key; display_name is what the farmer sees. original_label is the "
    "0-based IP102 label, project_label is this scope's own 0-based label. Slugs for "
    "labels also present in classes_top15.json are copied from that file so both "
    "scopes resolve to the same treatment guide."
)


def slugify(name: str) -> str:
    """Reduce a ``classes.txt`` name to a stable lowercase identifier."""
    cleaned = re.sub(r"[^a-z0-9]+", "_", name.strip().lower())
    return cleaned.strip("_")


def read_classes_file(path: Path) -> dict[int, str]:
    """Parse ``classes.txt`` into ``{ip102_label: name}``.

    ``classes.txt`` numbers its rows from 1 while every manifest label is
    0-based, so row ``id`` describes IP102 label ``id - 1``. That offset is the
    single most repeatable mistake in this dataset, so it is applied here once
    and asserted against the row count.

    Raises:
        SystemExit: If the file is missing or malformed.
    """
    if not path.is_file():
        raise SystemExit(
            f"classes.txt not found at {path}.\n"
            f"This script needs the IP102 archive present. Its output is committed, "
            f"so a checkout without the dataset does not need to run it."
        )

    names: dict[int, str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(maxsplit=1)
        if len(parts) != 2 or not parts[0].isdigit():
            raise SystemExit(f"{path}:{number}: cannot parse class line {line!r}")
        one_based, name = int(parts[0]), parts[1].strip()
        if one_based - 1 in names:
            raise SystemExit(f"{path}:{number}: duplicate class id {one_based}")
        names[one_based - 1] = name
    if not names:
        raise SystemExit(f"{path} contains no classes")
    return names


def read_guided_slugs(path: Path) -> dict[int, dict[str, str]]:
    """Map IP102 label -> the slug and display name already used by the guides."""
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(entry["original_label"]): {
            "class_name": entry["class_name"],
            "display_name": entry.get("display_name", ""),
        }
        for entry in payload.get("classes", [])
    }


def build_table(
    scope: ScopeSpec, names: dict[int, str], guided: dict[int, dict[str, str]]
) -> dict[str, Any]:
    """Render one scope's class table.

    Raises:
        SystemExit: If the scope names a label ``classes.txt`` does not define,
            or if two classes reduce to the same slug -- either would make a
            prediction ambiguous rather than merely mislabelled.
    """
    entries: list[dict[str, Any]] = []
    seen: dict[str, int] = {}

    for project_label, original_label in enumerate(scope.original_labels):
        if original_label not in names:
            raise SystemExit(
                f"scope {scope.name!r} includes IP102 label {original_label}, which "
                f"classes.txt does not define"
            )
        source_name = names[original_label]
        borrowed = guided.get(original_label)
        slug = borrowed["class_name"] if borrowed else slugify(source_name)
        display = (
            borrowed["display_name"]
            if borrowed and borrowed["display_name"]
            else source_name[:1].upper() + source_name[1:]
        )
        if slug in seen:
            raise SystemExit(
                f"scope {scope.name!r}: labels {seen[slug]} and {original_label} both "
                f"reduce to slug {slug!r}; predictions would be indistinguishable"
            )
        seen[slug] = original_label
        entries.append(
            {
                "project_label": project_label,
                "original_label": original_label,
                "class_name": slug,
                "display_name": display,
                "ip102_name": source_name,
                # Whether app/treatment_guides.py has vetted guidance for this
                # pest. Recorded rather than computed at run time so the gap is
                # visible in the committed artifact.
                "has_treatment_guide": original_label in guided,
            }
        )

    return {
        "subset": scope.name,
        "source": "classification",
        "class_mapping_version": CLASS_MAPPING_VERSION,
        "description": DESCRIPTION.format(scope=scope.name),
        "num_classes": scope.num_classes,
        "classes_with_treatment_guide": sum(e["has_treatment_guide"] for e in entries),
        "classes": entries,
    }


def render(table: dict[str, Any]) -> str:
    """Serialise a table to the exact bytes written to disk."""
    return json.dumps(table, indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    """Render every scope's class table, or verify the committed copies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--classes-file",
        type=Path,
        default=DEFAULT_CLASSES_FILE,
        help="Path to IP102 classes.txt.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare against the committed files and exit non-zero on a difference.",
    )
    args = parser.parse_args()

    names = read_classes_file(args.classes_file)
    guided = read_guided_slugs(GUIDED_CLASSES_FILE)
    print(f"Read {len(names)} class names from {args.classes_file}")
    print(f"Treatment guides cover {len(guided)} IP102 labels\n")

    differences = 0
    for scope in SCOPES:
        table = build_table(scope, names, guided)
        content = render(table)
        destination = MANIFEST_DIR / f"classes_{scope.name}.json"
        covered = table["classes_with_treatment_guide"]

        if args.check:
            existing = (
                destination.read_text(encoding="utf-8") if destination.is_file() else ""
            )
            status = "OK" if existing == content else "DIFFERS"
            differences += existing != content
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content, encoding="utf-8")
            status = "written"

        print(
            f"  {scope.name:<8} {scope.num_classes:>3} classes, "
            f"{covered:>3} with treatment guidance  ->  "
            f"{destination.relative_to(PROJECT_ROOT)}  [{status}]"
        )

    if args.check and differences:
        print(f"\n{differences} file(s) differ from the committed copy.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
