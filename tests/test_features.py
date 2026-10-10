"""Feature builder tests on small hand-made tables."""

import pandas as pd
import pytest

from factorymind.features import FEATURES, NO_REPLACEMENT_HOURS, build_features, latest_features


def tables():
    hours = pd.date_range("2015-01-01 00:00", periods=30, freq="h")
    tel = pd.DataFrame({
        "datetime": list(hours) * 2,
        "machineID": [1] * 30 + [2] * 30,
        "volt": [170.0] * 60,
        "rotate": list(range(400, 430)) + [450.0] * 30,
        "pressure": [100.0] * 60,
        "vibration": [40.0] * 60,
    })
    errors = pd.DataFrame({
        "datetime": pd.to_datetime(["2015-01-01 02:00", "2015-01-01 05:00", "2015-01-02 04:00"]),
        "machineID": [1, 1, 1],
        "errorID": ["error2", "error2", "error3"],
    })
    maint = pd.DataFrame({
        "datetime": pd.to_datetime(["2014-12-31 00:00", "2015-01-01 10:00"]),
        "machineID": [1, 1],
        "comp": ["comp2", "comp2"],
    })
    machines = pd.DataFrame({"machineID": [1, 2], "model": ["model3", "model1"], "age": [18, 7]})
    return {"telemetry": tel, "errors": errors, "maint": maint, "machines": machines}


def at(features, machine, when):
    row = features[(features.machineID == machine) & (features.datetime == pd.Timestamp(when))]
    assert len(row) == 1
    return row.iloc[0]


def test_one_row_per_reading_with_all_features():
    f = build_features(tables())
    assert len(f) == 60
    assert list(f.columns) == ["machineID", "datetime", *FEATURES]
    assert not f[FEATURES].isna().any().any()


def test_rolling_means_use_only_past_readings():
    f = build_features(tables())
    # rotate on machine 1 is 400, 401, ... so the 3h mean at hour 5 is mean(403, 404, 405).
    assert at(f, 1, "2015-01-01 05:00")["rotate_mean_3h"] == pytest.approx(404.0)
    assert at(f, 1, "2015-01-01 00:00")["rotate_std_3h"] == 0.0


def test_error_counts_cover_the_last_24h():
    f = build_features(tables())
    assert at(f, 1, "2015-01-01 05:00")["error2_count_24h"] == 2
    # 02:00 drops out of the window 24h later; 05:00 is still in it.
    assert at(f, 1, "2015-01-02 02:00")["error2_count_24h"] == 1
    assert at(f, 1, "2015-01-02 04:00")["error3_count_24h"] == 1
    assert at(f, 2, "2015-01-01 05:00")["error2_count_24h"] == 0


def test_hours_since_replacement():
    f = build_features(tables())
    assert at(f, 1, "2015-01-01 09:00")["hours_since_comp2"] == 33
    assert at(f, 1, "2015-01-01 12:00")["hours_since_comp2"] == 2
    assert at(f, 2, "2015-01-01 12:00")["hours_since_comp2"] == NO_REPLACEMENT_HOURS


def test_machine_attributes():
    row = at(build_features(tables()), 1, "2015-01-01 00:00")
    assert (row["is_model3"], row["is_model1"], row["age"]) == (1, 0, 18)


def test_latest_features_gives_each_machines_last_reading():
    latest = latest_features(tables())
    assert list(latest.machineID) == [1, 2]
    assert (latest.datetime == pd.Timestamp("2015-01-02 05:00")).all()
