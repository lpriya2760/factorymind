"""Tests at the analysis-engine seam: analyze(files, assumptions, seed) -> AnalysisReport."""

import io
import json
import zipfile
from pathlib import Path

import pytest

from factorymind.bundle import MAX_BUNDLE_BYTES, BundleValidationError
from factorymind.engine import analyze

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_bundle"


@pytest.fixture
def files() -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(FIXTURE.glob("*.csv"))}


def errors_for(files, **kwargs):
    with pytest.raises(BundleValidationError) as exc:
        analyze(files, **kwargs)
    return exc.value.errors


def has_error(errors, table, column=None, text=""):
    return any(e.table == table and e.column == column and text in e.message for e in errors)


def edit(files, name, old, new):
    files[name] = files[name].replace(old.encode(), new.encode())
    return files


# --- valid bundles ---------------------------------------------------------------------------

def test_valid_bundle_lists_machines_sorted_by_id(files):
    report = analyze(files)
    assert [(m.machine_id, m.model, m.age) for m in report.machines] == [
        (1, "model3", 18),
        (2, "model4", 7),
    ]


def test_accepts_kaggle_file_names(files):
    renamed = {f"PdM_{name}": data for name, data in files.items()
               if name in {"telemetry.csv", "errors.csv", "maint.csv", "machines.csv"}}
    renamed |= {name: data for name, data in files.items() if name not in {
        "telemetry.csv", "errors.csv", "maint.csv", "machines.csv"}}
    assert len(analyze(renamed).machines) == 2


def test_accepts_a_zipped_bundle(files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(f"demo_bundle/{name}", data)
        zf.writestr("__MACOSX/demo_bundle/._telemetry.csv", b"junk")
    assert len(analyze({"demo_bundle.zip": buf.getvalue()}).machines) == 2


def test_seed_is_echoed(files):
    assert analyze(files, seed=42).seed == 42


def test_report_serialises_to_plain_dict(files):
    d = analyze(files).to_dict()
    assert d["machines"][0] == {"machine_id": 1, "model": "model3", "age": 18}


# --- validation errors -----------------------------------------------------------------------

def test_missing_file(files):
    del files["orders.csv"]
    assert has_error(errors_for(files), "orders", None, "missing file")


def test_missing_column(files):
    edit(files, "telemetry.csv", "vibration", "vibe")
    assert has_error(errors_for(files), "telemetry", "vibration", "missing column")


def test_wrong_type_reports_line_and_value(files):
    edit(files, "telemetry.csv", "402.7", "fast")
    assert has_error(errors_for(files), "telemetry", "rotate", "line 3: 'fast'")


def test_non_integer_in_whole_number_column(files):
    edit(files, "machines.csv", "model3,18", "model3,18.5")
    assert has_error(errors_for(files), "machines", "age", "not whole numbers")


def test_bad_date(files):
    edit(files, "orders.csv", "2015-01-03", "next week")
    assert has_error(errors_for(files), "orders", "due_date", "not dates")


def test_empty_value(files):
    edit(files, "inventory.csv", "P-200,300,100", "P-200,,100")
    assert has_error(errors_for(files), "inventory", "on_hand", "empty value")


def test_unknown_machine_ids(files):
    edit(files, "telemetry.csv", "07:00:00,2,", "07:00:00,99,")
    edit(files, "production_schedule.csv", "2,P-200", "7,P-200")
    errors = errors_for(files)
    assert has_error(errors, "telemetry", "machineID", "not in machines.csv: 99")
    assert has_error(errors, "production_schedule", "machine_id", "not in machines.csv: 7")


def test_duplicate_machine_ids(files):
    edit(files, "machines.csv", "2,model4,7", "1,model4,7")
    assert has_error(errors_for(files), "machines", "machineID", "duplicate")


def test_unrecognised_file(files):
    files["telemtry.csv"] = files.pop("telemetry.csv")
    errors = errors_for(files)
    assert has_error(errors, "bundle", None, "unrecognised file 'telemtry.csv'")


def test_oversized_upload_is_rejected_before_parsing(files):
    files["telemetry.csv"] += b"x" * MAX_BUNDLE_BYTES
    errors = errors_for(files)
    assert len(errors) == 1 and has_error(errors, "bundle", None, "limit")


def test_reports_every_problem_at_once(files):
    del files["orders.csv"]
    edit(files, "telemetry.csv", "vibration", "vibe")
    edit(files, "inventory.csv", "P-200,300,100", "P-200,,100")
    assert len(errors_for(files)) == 3


def test_invalid_zip(files):
    assert has_error(errors_for({"bundle.zip": b"not a zip"}), "bundle", None, "not a valid zip")


# --- provenance ------------------------------------------------------------------------------

def test_default_provenance(files):
    p = analyze(files).data_provenance
    assert p["telemetry"] == "real" and p["machines"] == "real"
    assert p["orders"] == "synthetic" and p["inventory"] == "synthetic"


def test_manifest_overrides_provenance(files):
    files["manifest.json"] = json.dumps({"provenance": {"orders": "real"}}).encode()
    p = analyze(files).data_provenance
    assert p["orders"] == "real" and p["inventory"] == "synthetic"


def test_bad_manifest(files):
    files["manifest.json"] = json.dumps({"provenance": {"orders": "made-up"}}).encode()
    assert has_error(errors_for(files), "manifest.json", None, "'real' or 'synthetic'")


# --- assumptions -----------------------------------------------------------------------------

def test_default_assumptions_are_reported(files):
    assert analyze(files).assumptions_used["downtime_cost_per_hour"] > 0


def test_assumption_overrides(files):
    used = analyze(files, assumptions={"downtime_cost_per_hour": 5000}).assumptions_used
    assert used["downtime_cost_per_hour"] == 5000


@pytest.mark.parametrize("bad", [{"nonsense": 1}, {"repair_hours": -1}])
def test_bad_assumptions_raise(files, bad):
    with pytest.raises(ValueError):
        analyze(files, assumptions=bad)
