from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from UI.widgets.measurements import dendroelevator_io
from UI.widgets.measurements.measurement_io import (
    DeserializeResult,
    ExportResult,
    FormatContext,
    MeasurementLike,
    deserialize_measurements,
    is_fieldweave_document,
    serialize_measurements,
)
from UI.widgets.measurements.measurement_kind import MeasurementKindRegistry


@dataclass(frozen=True)
class MeasurementFormat:
    """One measurement file format FieldWeave can read, and optionally write — see detect_format for how a file is matched to one."""

    key: str
    label: str
    file_filter: str
    sniff: Callable[[object], bool]
    load: Callable[[dict, FormatContext, MeasurementKindRegistry], DeserializeResult]
    save: Callable[[list[MeasurementLike], FormatContext], ExportResult]
    indent: int | None = None


FIELDWEAVE = MeasurementFormat(
    key="fieldweave",
    label="FieldWeave measurements",
    file_filter="FieldWeave measurements (*.json)",
    sniff=is_fieldweave_document,
    load=lambda doc, ctx, registry: deserialize_measurements(doc, registry),
    save=lambda measurements, ctx: ExportResult(serialize_measurements(measurements)),
    indent=2,
)

DENDROELEVATOR = MeasurementFormat(
    key="dendroelevator",
    label="Dendroelevator",
    file_filter="Dendroelevator (*.json)",
    sniff=dendroelevator_io.sniff,
    load=dendroelevator_io.load,
    save=dendroelevator_io.save,
)

# FieldWeave's own format is checked first: it is the only one that names
# itself, so a file that matches it is never mistaken for a foreign one.
FORMATS: tuple[MeasurementFormat, ...] = (FIELDWEAVE, DENDROELEVATOR)


def detect_format(document: object) -> MeasurementFormat | None:
    return next((fmt for fmt in FORMATS if fmt.sniff(document)), None)


def format_for_filter(file_filter: str) -> MeasurementFormat:
    """The format a save dialog's selected filter names, defaulting to FieldWeave's own."""
    return next((fmt for fmt in FORMATS if fmt.file_filter == file_filter), FIELDWEAVE)


def read_measurement_file(path: str | Path, registry: MeasurementKindRegistry, ctx: FormatContext) -> DeserializeResult:
    # utf-8-sig: Dendroelevator's files begin with a byte-order mark.
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        return DeserializeResult(warnings=[f"Could not read {path}: {exc}"])
    fmt = detect_format(document)
    if fmt is None:
        return DeserializeResult(warnings=[f"{Path(path).name} is not a recognized measurement file."])
    return fmt.load(document, ctx, registry)


def write_measurement_file(
    path: str | Path, fmt: MeasurementFormat, measurements: list[MeasurementLike], ctx: FormatContext
) -> ExportResult:
    """Write *measurements* in *fmt*. Nothing is written (and ``document`` is None) when the format can't represent them — the warnings say why."""
    result = fmt.save(measurements, ctx)
    if result.document is None:
        return result
    try:
        Path(path).write_text(json.dumps(result.document, indent=fmt.indent), encoding="utf-8")
    except OSError as exc:
        result.warnings.append(f"Could not write {path}: {exc}")
        result.document = None
    return result
