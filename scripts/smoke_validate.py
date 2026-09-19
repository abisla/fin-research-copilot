"""Validation-chain gate: numeric consistency -> citation coverage -> grounding.

Pure and offline — synthetic answers against synthetic evidence, no DB, no LLM — for the
same reason smoke_answer asserts the citation check on hand-built text: the property is
"a figure that is not in the evidence is caught", and waiting for a model to actually
invent one would make the gate flaky in the direction that hides bugs.

FC-14 is the anchor case: the answer cited $44.1B and $39.3B while the SQL evidence
held 96.22 / 81.61 / 68.13 / 57.01. It must fail closed, and the same evidence with the
correct figures must pass, otherwise the validator is just a refusal machine.

    python scripts/smoke_validate.py
"""
from unittest import mock

from src.generation import answer as A
from src.generation.prompts import INSUFFICIENT
from src.router import router as R

SQL = A.Evidence(
    key="financials:NVDA:revenue", kind="financial", label="NVDA · financials · metric_series(revenue)",
    text="FY2027Q2: $96.22B\nFY2027Q1: $81.61B\nFY2026Q4: $68.13B\nFY2026Q3: $57.01B")
FILING = A.Evidence(
    key="chunk-1", kind="filing", label="NVDA · 10-Q",
    text=("Revenue was $44,062 million for the quarter, up 12% from the prior quarter. "
          "Data Center compute revenue grew on demand for Blackwell accelerators. "
          "Gross margin was 73.4% and we returned $9.7 billion to shareholders."))
EV = [SQL, FILING]


def chain(text, ev=EV, refusal=False):
    vs = A.run_chain(text, ev, refusal=refusal)
    failed = next((v for v in vs if not v.passed), None)
    return failed.name if failed else None, vs


def expect(label, text, want, ev=EV, refusal=False):
    got, vs = chain(text, ev, refusal)
    assert got == want, f"{label}: expected {want!r}, got {got!r} — {[v.detail for v in vs]}"
    print(f"  ok  {label:46s} -> {got or 'pass'}")


def main() -> int:
    # --- figure extraction ------------------------------------------------
    figs = {f.raw: f for f in A.extract_figures("Up 12.5% to $96.22B, EPS $2.94, 3 segments in FY2027Q2, Item 2.02 of the 10-K")}
    checked = sorted(f.raw for f in figs.values() if f.checkable)
    assert checked == ["$2.94", "$96.22B", "12.5%"], checked
    print(f"extract ok   - counts, years, fiscal labels and Item numbers are not figures: {checked}")

    print("\nnumeric consistency")
    # FC-14, verbatim shape: cited [1], numbers invented.
    expect("FC-14: $44.1B / $39.3B against 96.22-57.01",
           "NVDA revenue was $44.1B in Q1 [1] and $39.3B in Q4 [1].", "numeric_consistency")
    expect("correct figures, exact", "Revenue was $96.22B [1] after $81.61B [1].", None)
    expect("rounding: $96.2B vs 96.22", "Revenue reached $96.2B in the latest quarter [1].", None)
    expect("rounding: $96B vs 96.22", "Revenue reached about $96B in the latest quarter [1].", None)
    expect("unit conversion: $96,221 million", "Revenue was $96,221 million [1].", None)
    expect("$44.1B == $44,062 million in a filing", "Revenue was $44.1 billion for the quarter [2].", None)
    expect("percent in evidence", "Gross margin was 73.4% [2].", None)
    expect("bare table figure, any scale", "Revenue was 44,062 [2] and $44.1B [2].", None)
    expect("near miss is still wrong: $96.9B", "Revenue reached $96.9B [1].", "numeric_consistency")
    expect("derived growth rate is rejected", "Revenue grew 68.8% over four quarters [1].", "numeric_consistency")
    expect("wrong scale: $96.22M", "Revenue was $96.22M [1].", "numeric_consistency")
    expect("a range checks both ends", "Revenue was $57.01B to $60.0B [1].", "numeric_consistency")
    got, _ = chain(f"{INSUFFICIENT}: not stated, though revenue was about $12.9B.", refusal=True)
    assert got == "numeric_consistency", got
    print(f"  ok  {'refusal smuggling a figure (FC-12)':46s} -> {got}")
    expect("clean refusal passes", f"{INSUFFICIENT}: no 2030 dividend policy is disclosed.", None, refusal=True)

    print("\ncitation coverage")
    good = "Facts\n- Revenue was $96.22B [1].\n- Data Center compute revenue grew on Blackwell demand [2].\n"
    expect("fully cited Facts", good, None)
    expect("uncited figure sentence", "Facts\n- Revenue was $96.22B.\n- Revenue grew on Blackwell demand [2].",
           "citation_coverage")
    expect("no citations at all", "Data Center compute revenue grew on Blackwell demand from customers.",
           "citation_coverage")
    expect("trailing citation after full stop", "Data Center compute revenue grew on Blackwell demand. [2]", None)
    expect("header / short fragment are not claims", "Facts\n- Revenue was $96.22B [1].\n\nInterpretation\n- None.", None)
    expect("'no interpretation' line is allowed",
           "Facts\n- Revenue was $96.22B [1].\n\nInterpretation\n- The context supports no interpretation of the cause.", None)
    expect("stripped hallucinated [9] leaves the claim uncited",
           "Revenue was $96.22B [1]. Data Center compute revenue grew on Blackwell demand [9].".replace("[9]", ""),
           "citation_coverage")

    # Regressions found calibrating against real llama3.1 output (see DECISIONS #32).
    expect("grouped citation [1, 2] counts", "Data Center compute revenue grew on Blackwell demand [1, 2].", None)
    clean, cited, halluc = A.check_citations("Grew on demand [1, 9] per the filing [2, 8].", 3)
    assert cited == [1, 2] and halluc == [8, 9] and "9" not in clean and "8" not in clean, (clean, cited, halluc)
    assert "[1]" in clean and "[2]" in clean, f"valid members of a group must survive: {clean!r}"
    print(f"  ok  {'hallucinated member stripped from a group':46s} -> {clean!r}")
    expect("'J.P. Morgan' is one sentence, not two",
           "Data Center compute revenue grew as J.P. Morgan noted the Blackwell demand [2].", None)
    table = A.Evidence(key="t", kind="filing", label="NVDA · 10-Q · Financial Statements",
                       text="Cost of revenue\n$\n71\n$\n58\nResearch and development\n1,551\n1,191")
    expect("table cell '$ 71' is in millions", "Cost of revenue was $71 million [1].", None, ev=[table])
    expect("absence statement needs no citation",
           "The CFO did not explicitly mention the 2030 dividend policy in any of the provided documents.", None)

    expect("'couldn't find ... in the provided context'",
           "I couldn't find any information about NVIDIA's statements on export restrictions in the provided context.", None)
    expect("absence exemption does not launder a claim",
           "Revenue was $96.22B, which the provided context confirms.", "citation_coverage")

    print("\ngrounding")
    expect("figure pinned to the wrong item", "Revenue was $96.22B [2].", "grounding")
    expect("claim the cited item never makes",
           "Management expects tariffs to compress European automotive margins [2].", "grounding")
    expect("paraphrase with overlap passes", "Data Center compute revenue grew on demand for Blackwell accelerators [2].", None)

    print("\nfail closed through generate()")
    ctx = R.classify("Compare NVDA revenue", use_llm=False)
    fc14 = "NVDA revenue was $44.1B in Q1 [1] and $39.3B in Q4 [1]."
    with mock.patch.object(A, "available", return_value=True), \
         mock.patch.object(A, "complete", return_value=fc14):
        ans = A.generate("Compare NVDA revenue", EV, ctx)
    assert ans.insufficient_evidence and ans.text.startswith(INSUFFICIENT), ans.text
    assert "$44.1B" not in ans.text and "$39.3B" not in ans.text, "refusal text repeats the invented figures"
    assert ans.citations == [] and ans.rejected_text == fc14
    assert ans.validation and ans.validation[-1].name == "numeric_consistency" and not ans.validation[-1].passed
    print(f"  ok  FC-14 answer withheld -> {ans.text!r}")

    ok = "Revenue was $96.22B [1] and $81.61B [1]."
    with mock.patch.object(A, "available", return_value=True), \
         mock.patch.object(A, "complete", return_value=ok):
        ans = A.generate("Compare NVDA revenue", EV, ctx)
        off = A.generate("Compare NVDA revenue", EV, ctx, validate=False)
        with mock.patch.object(A, "complete", return_value=fc14):
            off_bad = A.generate("Compare NVDA revenue", EV, ctx, validate=False)
    assert not ans.insufficient_evidence and ans.citations == ["financials:NVDA:revenue"], ans
    assert [v.passed for v in ans.validation] == [True, True, True], ans.validation
    assert off.validation == [] and off_bad.text == fc14, "validate=False must be a true bypass (A/B arm)"
    print("  ok  correct answer passes all three, validate=False bypasses")

    print("\nPHASE 6b VALIDATION CHAIN PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
