"""Loading and validating an uploaded bundle of CSV files.

A bundle is the machine data (Azure PdM telemetry, errors, maintenance, machines) plus the
plant's business context (production schedule, inventory, orders), optionally zipped, with an
optional manifest.json declaring which tables are real and which are synthetic.
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Mapping

import pandas as pd

# Cloud Run caps HTTP/1 request bodies at 32 MiB; leave headroom for multipart overhead.
MAX_BUNDLE_BYTES = 30 * 1024 * 1024
# Guards against zip bombs: the total size of a zip's contents once extracted.
MAX_UNCOMPRESSED_BYTES = 256 * 1024 * 1024

MANIFEST = "manifest.json"
PROVENANCES = ("real", "synthetic")


@dataclass(frozen=True)
class TableSpec:
    columns: dict[str, str]  # column name -> "datetime" | "int" | "float" | "str"
    provenance: str  # assumed when the manifest doesn't say


TABLES: dict[str, TableSpec] = {
    "telemetry": TableSpec(
        {"datetime": "datetime", "machineID": "int", "volt": "float", "rotate": "float",
         "pressure": "float", "vibration": "float"},
        "real",
    ),
    "errors": TableSpec({"datetime": "datetime", "machineID": "int", "errorID": "str"}, "real"),
    "maint": TableSpec({"datetime": "datetime", "machineID": "int", "comp": "str"}, "real"),
    "machines": TableSpec({"machineID": "int", "model": "str", "age": "int"}, "real"),
    "production_schedule": TableSpec(
        {"machine_id": "int", "product": "str", "units_per_hour": "float"}, "synthetic"
    ),
    "inventory": TableSpec({"product": "str", "on_hand": "int", "safety_stock": "int"}, "synthetic"),
    "orders": TableSpec(
        {"order_id": "str", "customer": "str", "product": "str", "qty": "int",
         "due_date": "datetime", "penalty_per_day": "float", "priority": "str"},
        "synthetic",
    ),
}

# Tables whose machine IDs must all appear in the machines table.
MACHINE_ID_COLUMNS = {
    "telemetry": "machineID",
    "errors": "machineID",
    "maint": "machineID",
    "production_schedule": "machine_id",
}

# Alternative file names, e.g. the Kaggle download's "PdM_maint.csv" -> "maint".
_ALIASES = {"maintenance": "maint"}


@dataclass(frozen=True)
class FieldError:
    table: str  # a table name, or "bundle" for problems with the upload as a whole
    column: str | None
    message: str


class BundleValidationError(Exception):
    def __init__(self, errors: list[FieldError]):
        self.errors = errors
        super().__init__("; ".join(f"{e.table}.{e.column or '*'}: {e.message}" for e in errors))


@dataclass(frozen=True)
class Bundle:
    tables: dict[str, pd.DataFrame]
    provenance: dict[str, str]  # table name -> "real" | "synthetic"


def load_bundle(files: Mapping[str, bytes]) -> Bundle:
    """Parse and validate uploaded files (name -> contents).

    Raises BundleValidationError listing every problem found, so the user can fix them all at once.
    """
    upload_bytes = sum(len(b) for b in files.values())
    if upload_bytes > MAX_BUNDLE_BYTES:
        raise BundleValidationError([FieldError(
            "bundle", None,
            f"upload is {upload_bytes / 2**20:.1f} MB; the limit is {MAX_BUNDLE_BYTES / 2**20:.0f} MB. "
            "Upload fewer machines or a shorter time window.",
        )])

    errors: list[FieldError] = []
    named = _name_files(_expand_zips(files, errors), errors)
    if errors:
        raise BundleValidationError(errors)

    for name in TABLES:
        if name not in named:
            errors.append(FieldError(name, None, f"missing file {name}.csv"))
    provenance = _read_manifest(named.get(MANIFEST), errors)

    tables = {}
    for name, spec in TABLES.items():
        if name in named:
            df = _parse_table(name, spec, named[name], errors)
            if df is not None:
                tables[name] = df

    if "machines" in tables:
        _check_machine_ids(tables, errors)
    if errors:
        raise BundleValidationError(errors)
    return Bundle(tables=tables, provenance=provenance)


def _expand_zips(files: Mapping[str, bytes], errors: list[FieldError]) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    for name, data in files.items():
        if not name.lower().endswith(".zip"):
            out[name] = data
            continue
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                members = [m for m in zf.infolist()
                           if not m.is_dir() and not m.filename.startswith("__MACOSX/")]
                if sum(m.file_size for m in members) > MAX_UNCOMPRESSED_BYTES:
                    errors.append(FieldError("bundle", None, f"{name} is too large once extracted"))
                    continue
                for m in members:
                    out[m.filename] = zf.read(m)
        except zipfile.BadZipFile:
            errors.append(FieldError("bundle", None, f"{name} is not a valid zip file"))
    return out


def _name_files(files: Mapping[str, bytes], errors: list[FieldError]) -> dict[str, bytes]:
    """Map uploaded file names to table names (or MANIFEST)."""
    named: dict[str, bytes] = {}
    for filename, data in files.items():
        base = PurePosixPath(filename.replace("\\", "/")).name.lower()
        if base.startswith("."):
            continue  # e.g. .DS_Store inside a zip
        if base == MANIFEST:
            key = MANIFEST
        else:
            stem = base.removesuffix(".csv").removeprefix("pdm_")
            key = _ALIASES.get(stem, stem)
            if not base.endswith(".csv") or key not in TABLES:
                errors.append(FieldError("bundle", None, f"unrecognised file {filename!r}"))
                continue
        if key in named:
            errors.append(FieldError("bundle", None, f"more than one file for {key!r}"))
            continue
        named[key] = data
    return named


def _read_manifest(data: bytes | None, errors: list[FieldError]) -> dict[str, str]:
    provenance = {name: spec.provenance for name, spec in TABLES.items()}
    if data is None:
        return provenance
    try:
        declared = json.loads(data).get("provenance", {})
        if not isinstance(declared, dict):
            raise ValueError
    except (ValueError, AttributeError):
        errors.append(FieldError(MANIFEST, None,
                                 'must be JSON like {"provenance": {"orders": "synthetic"}}'))
        return provenance
    for table, value in declared.items():
        if table not in TABLES:
            errors.append(FieldError(MANIFEST, None, f"unknown table {table!r}"))
        elif value not in PROVENANCES:
            errors.append(FieldError(MANIFEST, None,
                                     f"provenance of {table!r} must be 'real' or 'synthetic'"))
        else:
            provenance[table] = value
    return provenance


def _parse_table(name: str, spec: TableSpec, data: bytes,
                 errors: list[FieldError]) -> pd.DataFrame | None:
    try:
        raw = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        errors.append(FieldError(name, None, f"could not be read as CSV ({exc})"))
        return None

    missing = [c for c in spec.columns if c not in raw.columns]
    for col in missing:
        errors.append(FieldError(name, col, "missing column"))
    if missing:
        return None

    df = pd.DataFrame(index=raw.index)
    ok = True
    for col, kind in spec.columns.items():
        text = raw[col].str.strip()
        empty = text == ""
        if empty.any():
            errors.append(FieldError(name, col,
                                     f"{empty.sum()} empty value(s), first on line {_line(empty)}"))
            ok = False
            continue
        parsed, bad = _coerce(text, kind)
        if bad.any():
            errors.append(FieldError(
                name, col,
                f"{bad.sum()} value(s) are not {_KIND_NAMES[kind]}, first on line {_line(bad)}: "
                f"{text[bad].iloc[0]!r}",
            ))
            ok = False
            continue
        df[col] = parsed
    return df if ok else None


_KIND_NAMES = {"datetime": "dates", "int": "whole numbers", "float": "numbers", "str": "text"}


def _coerce(text: pd.Series, kind: str) -> tuple[pd.Series, pd.Series]:
    """Return (parsed values, mask of values that failed to parse)."""
    if kind == "datetime":
        parsed = pd.to_datetime(text, errors="coerce", format="mixed")
        return parsed, parsed.isna()
    if kind in ("int", "float"):
        parsed = pd.to_numeric(text, errors="coerce")
        bad = parsed.isna()
        if kind == "int":
            bad |= parsed % 1 != 0
            parsed = parsed.where(~bad, 0).astype("int64")
        return parsed, bad
    return text, pd.Series(False, index=text.index)


def _line(mask: pd.Series) -> int:
    """CSV line number of the first True row (line 1 is the header)."""
    return int(mask.to_numpy().argmax()) + 2


def _check_machine_ids(tables: dict[str, pd.DataFrame], errors: list[FieldError]) -> None:
    ids = tables["machines"]["machineID"]
    dupes = sorted(ids[ids.duplicated()].unique())
    if dupes:
        errors.append(FieldError("machines", "machineID", f"duplicate machine IDs: {_list(dupes)}"))
    known = set(ids)
    for name, col in MACHINE_ID_COLUMNS.items():
        if name in tables:
            unknown = sorted(set(tables[name][col]) - known)
            if unknown:
                errors.append(FieldError(
                    name, col, f"machine IDs not in machines.csv: {_list(unknown)}"))


def _list(values: list, limit: int = 5) -> str:
    shown = ", ".join(str(v) for v in values[:limit])
    return shown + (f" and {len(values) - limit} more" if len(values) > limit else "")
