"""Known-answer label, feature and financial-join tests on synthetic histories."""

import json
import math
import sqlite3
from datetime import date, timedelta

import pytest

from mqf_research.data import build_abnormal_returns as ar
from mqf_research.data import build_market_features as market
from mqf_research.data import build_model_dataset as dataset


def synthetic_returns(count=40):
    """Create exact stock = alpha + beta * benchmark intraday-return observations."""
    stock, benchmark = {}, {}
    for index in range(count):
        day = (date(2020, 1, 1) + timedelta(days=index)).isoformat()
        movement = 0.01 * math.sin(index / 3)
        stock[day] = ar.Bar(day, 100.0, 100 * (1 + 0.001 + 1.5 * movement))
        benchmark[day] = ar.Bar(day, 200.0, 200 * (1 + movement))
    return stock, benchmark


def test_ols_known_coefficients_exclude_reaction_and_future():
    """Large outcome perturbations cannot change coefficients fit strictly before reaction."""
    stock, benchmark = synthetic_returns()
    reaction = "2020-02-10"
    before = ar.estimate_intraday_ols(stock, benchmark, reaction, window=20, minimum=20)
    stock[reaction] = ar.Bar(reaction, 1, 1000)
    benchmark[reaction] = ar.Bar(reaction, 1, 0.01)
    stock["2020-02-11"] = ar.Bar("2020-02-11", 1, 9000)
    benchmark["2020-02-11"] = ar.Bar("2020-02-11", 1, 0.001)
    after = ar.estimate_intraday_ols(stock, benchmark, reaction, window=20, minimum=20)
    assert after == before
    alpha, beta, n, start, end = after
    assert alpha == pytest.approx(0.001, abs=1e-12)
    assert beta == pytest.approx(1.5, abs=1e-12)
    assert n == 20
    assert start == "2020-01-21"
    assert end == "2020-02-09"


def test_ols_counts_valid_matched_sessions_without_forward_filling():
    """Missing or invalid pairs extend the lookback instead of becoming fabricated returns."""
    stock, benchmark = synthetic_returns()
    stock["2020-02-08"] = ar.Bar("2020-02-08", None, 100)
    del benchmark["2020-02-07"]
    alpha, beta, n, start, end = ar.estimate_intraday_ols(
        stock, benchmark, "2020-02-10", window=20, minimum=20
    )
    assert (n, start, end) == (20, "2020-01-19", "2020-02-09")
    assert alpha == pytest.approx(0.001, abs=1e-12)
    assert beta == pytest.approx(1.5, abs=1e-12)


def test_ols_returns_none_for_insufficient_data_and_zero_market_variance():
    """Undefined market-model fits stay unavailable rather than becoming zero beta."""
    stock, benchmark = synthetic_returns(count=5)
    assert ar.estimate_intraday_ols(stock, benchmark, "2021-01-01", 20, 20) is None
    benchmark = {day: ar.Bar(day, 100, 100) for day in benchmark}
    assert ar.estimate_intraday_ols(stock, benchmark, "2021-01-01", 5, 5) is None


@pytest.mark.parametrize("window,minimum", [(0, 0), (1, 1), (3, 4), (2.5, 2)])
def test_ols_rejects_invalid_configuration(window, minimum):
    """Impossible window settings fail before labels are produced."""
    with pytest.raises(ValueError, match="window >= minimum >= 2"):
        ar.estimate_intraday_ols({}, {}, "2020-01-01", window, minimum)


@pytest.mark.parametrize(
    "opening,close", [(0, 1), (-1, 1), (None, 1), (math.inf, 1), (1, math.nan)]
)
def test_target_return_rejects_nonprices(opening, close):
    """Invalid price inputs cannot create seemingly valid decimal-return labels."""
    assert ar.open_close_return(ar.Bar("2020-01-01", opening, close)) is None


def test_adjusted_atr_has_no_mechanical_split_jump():
    """A 2-for-1 raw-price split leaves the constant adjusted true-range feature unchanged."""
    bars = []
    for index in range(15):
        day = (date(2020, 1, 1) + timedelta(days=index)).isoformat()
        scale = 2 if index < 7 else 1
        bars.append(
            market.Bar(day, 50 * scale, 50.5 * scale, 49.5 * scale, 50 * scale, 50, 1_000_000)
        )
    bars.append(market.Bar("2020-01-16", 1, 10_000, 0.01, 9999, 9999, 100_000_000))
    assert market.normalized_atr_14(bars, "2020-01-16") == pytest.approx(0.02)


def test_pre_event_features_are_invariant_to_reaction_bar():
    """Momentum, volatility and volume use completed prior sessions only."""
    bars = []
    for index in range(21):
        day = (date(2020, 1, 1) + timedelta(days=index)).isoformat()
        value = 100 * 1.01**index
        bars.append(market.Bar(day, value, value, value, value, value, 1_000_000))
    reaction = "2020-01-22"
    bars.append(market.Bar(reaction, 1, 1000, 0.1, 800, 800, 10_000_000))
    assert market.momentum_20(bars, reaction) == pytest.approx(1.01**20 - 1)
    assert market.realized_vol_20(bars, reaction) == pytest.approx(0, abs=1e-14)
    assert market.volume_surprise(bars, reaction) == pytest.approx(0, abs=1e-12)
    assert market.prior_bars(bars, reaction, 21)[-1].date == "2020-01-21"


def test_duplicate_event_identity_is_rejected():
    """Duplicate event rows cannot silently overwrite each other during feature joins."""
    with pytest.raises(ValueError, match="duplicate event_id"):
        dataset.unique_map([{"event_id": "ABC:1"}, {"event_id": "ABC:1"}], "event_id", "fixture")


def test_loss_eps_surprise_uses_absolute_consensus_denominator():
    """A smaller-than-expected loss is a positive surprise, while zero consensus is flagged."""
    with sqlite3.connect(":memory:") as db:
        db.execute(
            "CREATE TABLE earnings_events (event_id TEXT, actual_eps REAL, consensus_eps REAL, release_timing TEXT, source_file TEXT, raw_event_json TEXT)"
        )
        db.executemany(
            "INSERT INTO earnings_events VALUES (?,?,?,?,?,?)",
            [
                (
                    "loss",
                    -0.4,
                    -0.5,
                    "AfterMarket",
                    "eodhd/earnings/ABC.US/2020.json",
                    json.dumps({"actual": -0.4, "estimate": -0.5}),
                ),
                (
                    "zero",
                    0.1,
                    0,
                    "AfterMarket",
                    "eodhd/earnings/ABC.US/2020.json",
                    json.dumps({"actual": 0.1, "estimate": 0}),
                ),
            ],
        )
        result = dataset.load_structured_from_db(db)
    assert result["loss"]["eps_surprise_abs"] == pytest.approx(0.1)
    assert result["loss"]["eps_surprise_pct"] == pytest.approx(0.2)
    assert result["loss"]["estimate_pit_status"] == "vendor_defined_pre_release"
    assert result["zero"]["eps_consensus_zero"] == 1
    assert result["zero"]["eps_surprise_pct"] is None


def test_target_builder_persists_known_answer_without_modifying_prices(tmp_path):
    """The full SQLite target builder reproduces the residual formula on a synthetic event."""
    warehouse = tmp_path / "warehouse"
    warehouse.mkdir()
    batch = tmp_path / "batch"
    event = batch / "ai_inputs" / "ABC" / "event"
    event.mkdir(parents=True)
    (batch / "DONE.json").write_text("{}", encoding="utf-8")
    reaction = "2021-01-04"
    manifest = {
        "event_id": "ABC:2021-01-01:2020-12-31",
        "ticker": "ABC",
        "release_date": "2021-01-01",
        "fiscal_end": "2020-12-31",
        "reaction_session": reaction,
        "packaging_policy_version": "actual_release_v1",
        "effective_release_timestamp_utc": "2021-01-01T21:05:00Z",
        "reaction_market_open_utc": "2021-01-04T14:30:00Z",
        "point_in_time_verified": True,
        "release_time_verified": True,
        "readiness_status": "READY",
        "missing": [],
        "documents": [{"file": "documents/D001.txt"}],
        "original_assets": [],
    }
    (event / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    stock, benchmark = synthetic_returns(count=300)
    price_rows = [("ABC", day, bar.open, bar.close) for day, bar in stock.items()]
    price_rows += [("IWV", day, bar.open, bar.close) for day, bar in benchmark.items()]
    price_rows += [("ABC", reaction, 100, 105), ("IWV", reaction, 200, 204)]
    with sqlite3.connect(warehouse / "global.sqlite") as db:
        db.execute(
            "CREATE TABLE market_daily (ticker TEXT, trade_date TEXT, open REAL, close REAL)"
        )
        db.executemany("INSERT INTO market_daily VALUES (?,?,?,?)", price_rows)
    targets, issues, summary = ar.build(warehouse, batch, {"ABC"})
    assert issues == []
    assert len(targets) == summary["targets_built"] == 1
    assert targets[0].abnormal_return == pytest.approx(0.05 - (0.001 + 1.5 * 0.02), abs=1e-12)
    assert targets[0].market_estimation_n == 252
    assert targets[0].market_estimation_end < reaction
    with sqlite3.connect(warehouse / "global.sqlite") as db:
        assert db.execute("SELECT COUNT(*) FROM market_daily").fetchone()[0] == len(price_rows)
        assert db.execute("SELECT abnormal_return FROM targets_1d_abnormal").fetchone()[
            0
        ] == pytest.approx(targets[0].abnormal_return)
