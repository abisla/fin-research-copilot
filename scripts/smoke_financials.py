"""Phase 4 gate: structured financials and the canned query layer, on real data.

Read-only. Asserts the things that would otherwise fail silently and plausibly:
period classification, the Q4 derivation, tag selection, the gap-aware growth
arithmetic, and that a metric a filer does not report stays absent instead of
becoming zero.

    python scripts/smoke_financials.py
"""
from src.common.db import get_conn
from src.structured import guidance, queries
from src.structured.financials import ADDITIVE, fiscal_period, period_kind


def main():
    conn = get_conn()
    try:
        # --- period derivation is the foundation everything else sits on -----
        from datetime import date
        assert period_kind(date(2025, 7, 28), date(2025, 10, 26)) == "Q"
        assert period_kind(date(2025, 1, 27), date(2025, 10, 26)) == "9M"
        assert period_kind(date(2025, 1, 27), date(2026, 1, 25)) == "FY"
        assert period_kind(date(2025, 1, 27), date(2025, 2, 25)) is None      # monthly: ignored
        # offset fiscal calendars, verified against each filer's own labels
        assert fiscal_period(date(2025, 10, 26), 1) == (2026, 3), "NVDA FY2026Q3"
        assert fiscal_period(date(2026, 1, 25), 1) == (2026, 4), "NVDA FY2026 year end"
        assert fiscal_period(date(2025, 9, 30), 6) == (2026, 1), "MSFT FY2026Q1"
        assert fiscal_period(date(2025, 3, 31), 6) == (2025, 3), "MSFT FY2025Q3"
        assert fiscal_period(date(2025, 9, 30), 12) == (2025, 3), "JPM FY2025Q3"
        print("periods ok - duration classes + three different fiscal calendars")

        # --- coverage --------------------------------------------------------
        cov = queries.coverage(conn)
        assert cov, "financials is empty — run scripts/load_financials.py"
        print(f"\n{'ticker':7s} {'metric':18s} {'rows':>5s} {'derived':>8s}  years")
        for c in cov:
            print(f"{c['ticker']:7s} {c['metric']:18s} {c['n']:5d} {c['derived']:8d}  "
                  f"FY{c['from_fy']}-FY{c['to_fy']}")

        # --- a bank genuinely has no gross profit or operating income --------
        jpm = queries.available_metrics(conn, "JPM")
        nvda = queries.available_metrics(conn, "NVDA")
        assert "gross_margin" not in jpm and "operating_income" not in jpm, jpm
        assert "revenue" in jpm and "eps_diluted" in jpm, jpm
        assert {"revenue", "eps_diluted", "operating_income", "gross_margin"} <= set(nvda)
        assert queries.margin_trend(conn, "JPM") == [], "JPM must have no margin series"
        print(f"\nmetrics ok - JPM reports {jpm} (no gross profit / operating income: "
              f"a bank has no cost of revenue); NVDA reports {nvda}")

        # --- values and provenance -------------------------------------------
        rev = queries.revenue_last_n_quarters(conn, "NVDA", 6)
        assert len(rev) == 6 and all(p.metric == "revenue" for p in rev)
        assert all(p.fiscal_qtr > 0 for p in rev), "annual row leaked into a quarterly series"
        assert rev == sorted(rev, key=lambda p: p.key, reverse=True), "not newest-first"
        assert all(isinstance(p.value, float) for p in rev), "value must be float on both backends"
        assert all(p.source_accn for p in rev), "every row needs provenance"
        print(f"\nNVDA revenue: {'  '.join(f'{p.period}={p.format()}' for p in rev)}")

        # the derived Q4, checked against the arithmetic it claims to be
        q4 = next(p for p in queries.metric_series(conn, "NVDA", "revenue", 12)
                  if p.key == (2026, 4))
        assert q4.derived, "NVDA FY2026Q4 should be flagged derived (filer reports no Q4)"
        assert abs(q4.value - (215_938_000_000 - 147_811_000_000)) < 1e6, q4.value
        assert not next(p for p in rev if p.key == (2026, 3)).derived, "Q3 is filed, not derived"
        print(f"derived ok - FY2026Q4 {q4.format()} = FY total 215.938B - 9M 147.811B, "
              f"flagged derived, attributed to {q4.source_accn}")

        # EPS is excluded from Q4 derivation on purpose
        assert "eps_diluted" not in ADDITIVE
        eps_q4 = [p for p in queries.metric_series(conn, "NVDA", "eps_diluted", 20)
                  if p.key == (2026, 4)]
        assert not eps_q4, "Q4 EPS must not be derived — diluted share count moves"
        print("           EPS Q4 correctly absent (annual EPS != sum of quarterly EPS)")

        # --- the gap trap ----------------------------------------------------
        # FY2027Q1 EPS has no prior quarter in the table (Q4 EPS is never derived), so
        # QoQ must report "not comparable" rather than silently comparing to FY2026Q3.
        qoq = queries.qoq_growth(conn, "NVDA", "eps_diluted", 4)
        by_period = {g.point.key: g for g in qoq}
        gap = by_period.get((2027, 1))
        assert gap is not None, "expected FY2027Q1 EPS in the last 4 quarters"
        assert not gap.comparable and gap.pct_change is None, \
            "QoQ across the missing Q4 must be None, not a number"
        assert queries.prev_quarter(2027, 1) == (2026, 4)
        print(f"\ngap ok     - {gap.format()}")
        print("           (naive LIMIT-4 pairing would have compared FY2027Q1 to FY2026Q3)")

        # YoY over a contiguous stretch must compute
        yoy = {g.point.key: g for g in queries.yoy_growth(conn, "NVDA", "revenue", 4)}
        g = yoy[(2027, 2)]
        assert g.comparable and g.prior.key == (2026, 2), g.prior
        expected = 100.0 * (g.point.value - g.prior.value) / g.prior.value
        assert abs(g.pct_change - expected) < 1e-9
        assert g.pct_change > 50, g.pct_change
        print(f"yoy ok     - {g.format()}")
        for t in ("MSFT", "JPM"):
            gg = [x for x in queries.yoy_growth(conn, t, "revenue", 4) if x.comparable]
            assert gg, f"{t} has no comparable YoY growth"
            print(f"           {t}: {gg[0].format()}")

        # --- cross-ticker ----------------------------------------------------
        cmp = queries.compare_tickers(conn, ["NVDA", "MSFT", "JPM"], "revenue", 1)
        assert set(cmp) == {"NVDA", "MSFT", "JPM"} and all(v for v in cmp.values())
        print("\ncompare ok - latest revenue: " + ", ".join(
            f"{t} {p[0].period} {p[0].format()} (ended {p[0].period_end})"
            for t, p in cmp.items()))
        print("           fiscal calendars differ — period_end is why the answer can date it")

        # --- the registry the router drives ----------------------------------
        assert set(queries.QUERIES) >= {"revenue_last_n_quarters", "yoy_growth", "margin_trend"}
        out = queries.run(conn, "revenue_last_n_quarters", ticker="MSFT", n_quarters=2)
        assert len(out) == 2 and out[0].ticker == "MSFT"
        for bad, kwargs in (("drop_table", {}), ("yoy_growth", {"ticker": "NVDA", "sql": "x"})):
            try:
                queries.run(conn, bad, **kwargs)
                raise AssertionError(f"run() accepted {bad} {kwargs}")
            except ValueError:
                pass
        print(f"\nregistry ok - {len(queries.QUERIES)} canned queries; unknown name and "
              f"unknown param both rejected (LLM never writes SQL)")

        # --- guidance --------------------------------------------------------
        nv = guidance.latest_guidance(conn, "NVDA", 1)
        assert nv, "no NVDA guidance extracted"
        assert nv[0]["fiscal_period"].startswith("FY") and "Q" in nv[0]["fiscal_period"]
        assert guidance.parse_period(
            "outlook for the third quarter of fiscal 2027 is as follows") == "FY2027Q3"
        assert guidance.extract_from_text("no forward looking statements here") is None
        assert guidance.latest_guidance(conn, "JPM") == [], \
            "JPM issues no press-release guidance — must stay empty, not fabricated"
        print(f"\nguidance ok - NVDA guides {nv[0]['fiscal_period']} "
              f"(given {nv[0]['given_on']}, src {nv[0]['source_doc_id']}); JPM has none")

        print("\nPHASE 4 STRUCTURED FINANCIALS PASSED")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
