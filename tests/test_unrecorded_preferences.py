"""Unrecorded preferences: a Trade Agreement in force while TRAINS records no preference."""

import math

import polars as pl

from exportspredictor import tariffs


def _rows(rows: list[tuple]) -> pl.DataFrame:
    """Feature rows: exporter, importer, hs6, Applied Tariff (percent), MFN (percent), agreement, customs union."""
    return pl.DataFrame(
        [(e, i, h, 2022, None if a is None else math.log1p(a / 100), m, ta, cu) for e, i, h, a, m, ta, cu in rows],
        schema={"exporter": pl.String, "importer": pl.String, "hs6": pl.String, "year": pl.Int16,
                "log_applied_tariff": pl.Float64, "mfn_tariff": pl.Float64,
                "trade_agreement": pl.Int8, "customs_union": pl.Int8},
        orient="row",
    )


def test_an_agreement_partner_with_no_line_below_mfn_has_its_tariffs_set_to_zero():
    rows = _rows([
        ("GBR", "CAN", "220830", 5.0, 5.0, 1, 0),  # agreement, MFN charged on every line: unrecorded
        ("GBR", "CAN", "870323", 6.1, 6.1, 1, 0),
        ("GBR", "CAN", "300490", 0.0, 0.0, 1, 0),  # duty-free under MFN anyway
        ("GBR", "CAN", "851712", None, None, 1, 0),  # no schedule for the line: left missing
        ("GBR", "JPN", "220830", 0.0, 2.0, 1, 0),  # a preference recorded on one line ...
        ("GBR", "JPN", "870323", 3.0, 3.0, 1, 0),  # ... so the agreement's other lines are TRAINS' own
        ("GBR", "IND", "220830", 150.0, 150.0, 0, 0),  # no agreement: MFN is right
        ("DEU", "FRA", "220830", 0.0, 4.0, 1, 1),  # Customs Union: already 0
    ])

    out = tariffs.zero_unrecorded_preferences(rows)

    by_pair = {(r["importer"], r["hs6"]): r for r in out.iter_rows(named=True)}
    assert by_pair[("CAN", "220830")]["log_applied_tariff"] == 0.0
    assert by_pair[("CAN", "870323")]["log_applied_tariff"] == 0.0
    assert by_pair[("CAN", "851712")]["log_applied_tariff"] is None
    assert by_pair[("JPN", "870323")]["log_applied_tariff"] == math.log1p(0.03)
    assert by_pair[("IND", "220830")]["log_applied_tariff"] == math.log1p(1.5)
    assert out["unrecorded_preference"].to_list() == [True, True, True, True, False, False, False, False]
    assert out.columns == [*rows.columns, "unrecorded_preference"]
