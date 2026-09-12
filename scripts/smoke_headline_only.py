"""Gate: a headline-only cluster must not produce claims beyond its headlines.

This is the FC-6 regression test. 176 of 216 collected articles are Google News
redirects whose bodies are unrecoverable, so most clusters reach the summarizer as
headlines alone — and a headline *reads* like sufficient context, which is why the
model happily filled the rest in from parametric knowledge about the company. The
pre-fix brief (evals/before_realtext_NVDA.md) claimed analysts "cite its dominance
in the AI and gaming markets" and attributed results to "a diversified product
portfolio", from two headlines and nothing else, while citing [1] and [2] — which
makes invented content look sourced. That is worse than an empty answer.

The fixture is deliberately bait: three headlines that state *what* happened and
never *why*, so any cause, attribution or figure in the output is fabricated by
construction. Checks are deterministic string/regex properties rather than an
LLM judge, because a gate has to fail a build, and because the judge would have the
same blind spot as the generator.

Generation is stochastic, so the gate runs TRIALS times and every trial must pass.

    python scripts/smoke_headline_only.py
    python scripts/smoke_headline_only.py --trials 5
"""
import argparse
import re
import sys
from datetime import datetime, timezone

from src.generation import llm
from src.generation.prompts import NO_SUPPORT
from src.news import brief

TRIALS = 3

# Headlines state what happened, never why. No causal clause, one figure ("4").
FIXTURE = [
    ("Nvidia Announces 4 New AI Factory Sites", "Reuters", 8),
    ("Nvidia Expands Its AI Factory Footprint", "CNBC", 9),
    ("Nvidia Shares Move After Rack-Scale System Launch", "Barron's", 10),
]

# Attributing a view to a group none of the headlines quote.
ATTRIBUTION_RE = re.compile(
    r"\b(analysts?|investors?|experts?|traders?|observers?|the market|shareholders?)\b"
    r"[^.]{0,40}?\b(say|says|said|expect|expects|believe|believes|cite|cites|note|notes|"
    r"argue|argues|point to|see|sees|view|views|viewed|worry|worries|fear|fears)\b", re.I)

# Asserting a cause. None of the fixture headlines gives one.
CAUSAL_RE = re.compile(
    r"\b(because of|because|due to|driven by|thanks to|as a result of|owing to|"
    r"on the back of|attributed to|stems from|reflects|signals|indicates that)\b", re.I)

# Projection and passive attribution — the fabrication class the first draft of this
# gate missed entirely. The pre-fix model rarely wrote "analysts say"; it wrote "is
# seen as a strategic expansion", "are expected to support", "is likely to have a
# positive impact". Agentless hedging reads as analysis while asserting things no
# headline contains, so for a headline-only cluster it is unsupported by construction.
SPECULATION_RE = re.compile(
    r"\b(?:is|are|was|were|will be|would be)\s+(?:widely\s+|generally\s+)?"
    r"(?:expected|likely|seen|viewed|projected|anticipated|poised|set|believed|"
    r"positioned|considered)\s+(?:to|as)\b"
    r"|\b(?:is|are)\s+likely\b"
    r"|\b(?:may|might|could|should|will)\s+(?:lead to|result in|drive|boost|hurt|"
    r"benefit|pressure|view|see)\b"
    r"|\b(?:potentially\s+)?(?:leading|pointing)\s+to\b"
    r"|\b(?:suggests?|demonstrates?|highlights?|underscores?|confirms?)\b", re.I)

CITATION_RE = re.compile(r"\[(\d+)\]")


def build_event() -> brief.Event:
    """Three headline-only articles in one cluster. text=None is the whole point."""
    return brief.Event(event_id="fc3-fixture", ticker="NVDA", articles=[
        brief.Article(article_id=f"h{i}", headline=h, source=src,
                      published_at=datetime(2026, 9, day, 12, 0, tzinfo=timezone.utc),
                      url=f"https://news.google.com/rss/articles/CBMi{i}",
                      topic=None, is_duplicate=False, text=None)
        for i, (h, src, day) in enumerate(FIXTURE)])


def allowed_numbers(event: brief.Event) -> set[str]:
    """Digits the model may legitimately emit: from the headlines, from the article
    dates it is given, the citation indices, and the article count."""
    ok = set(re.findall(r"\d+", " ".join(a.headline for a in event.live)))
    for a in event.live:
        d = a.published_at
        ok |= {str(d.year), str(d.month), str(d.day), f"{d.month:02d}", f"{d.day:02d}"}
    ok |= {str(i) for i in range(1, len(event.live) + 1)}
    ok.add(str(len(event.live)))
    return ok


# A sentence that talks about the *evidence* rather than about the company. The model
# writing "details are not available due to the headline-only nature of the articles"
# is doing exactly what we asked, so the "due to" in it is not an asserted cause.
# Scoped to the sentence, so a real claim sitting next to a disclaimer still fails.
META_RE = re.compile(
    r"headline[- ]only|full[- ]text|full article|available headlines|"
    r"not (?:available|specified|provided|given|disclosed)|"
    r"no (?:body|further|additional) (?:text|detail)|based on the headlines?", re.I)

SENTENCE_RE = re.compile(r"[^.!?\n]*[.!?\n]|[^.!?\n]+$")


def _sentence_at(text: str, pos: int) -> str:
    for m in SENTENCE_RE.finditer(text):
        if m.start() <= pos < m.end():
            return m.group(0)
    return text


STOPWORDS = frozenset(
    "a an and are as at be but by for from has have in is it its of on or that the to "
    "was were will with this these those there here he she they them their according "
    "says said also into over under after before about".split())

GROUNDING = 0.7         # share of a sentence's content words that must come from one headline


def _content(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in STOPWORDS}


def _restates_a_headline(sentence: str, event: brief.Event) -> bool:
    """True when the sentence is substantially a restatement of one headline.

    Some headlines are themselves predictions ("Prediction: September 10 Will Be a Big
    Day for Nvidia Shareholders"), so a faithful restatement trips the projection
    pattern on its connective while claiming nothing new. Grounding is measured on the
    sentence's *content words*, not on whether it carries a citation — the original
    FC-3 fabrication cited [1] and [2], which is precisely how invention passed for
    sourced. Only overlap with the actual headline text earns the exemption.
    """
    words = _content(sentence)
    if not words:
        return True
    return any(len(words & _content(a.headline)) / len(words) >= GROUNDING
               for a in event.live)


def _flag(regex, label: str, summary: str, event: brief.Event) -> list[str]:
    """Matches in the summary that its headlines don't support.

    Three exemptions keep these patterns honest. If a headline itself says "analysts
    expect", repeating it is reporting. If the match sits in a sentence about the
    evidence ("details are not available due to the headline-only nature"), the model
    is disclaiming its limits — the behaviour this gate exists to encourage. And if the
    sentence simply restates a headline, the connective is not a claim of its own.
    """
    low = " ".join(a.headline for a in event.live).lower()
    out = []
    for m in regex.finditer(summary):
        sentence = _sentence_at(summary, m.start())
        if (m.group(0).lower() in low or META_RE.search(sentence)
                or _restates_a_headline(sentence, event)):
            continue
        out.append(f"{label}: {m.group(0)!r}")
    return out


def violations(summary: str, event: brief.Event) -> list[str]:
    """Every way this summary claims more than its headlines support."""
    out = []

    # 1. Fabricated figures. Strip citations first so [1] isn't read as a number.
    stripped = CITATION_RE.sub(" ", summary)
    ok = allowed_numbers(event)
    invented = sorted({n for n in re.findall(r"\d+", stripped) if n not in ok})
    if invented:
        out.append(f"invented figure(s) {invented} absent from every headline")

    # 2-4. Attribution, asserted cause, and projection dressed as analysis.
    out += _flag(ATTRIBUTION_RE, "unsupported attribution", summary, event)
    out += _flag(CAUSAL_RE, "asserted cause/inference", summary, event)
    out += _flag(SPECULATION_RE, "unsupported projection", summary, event)

    # 5. Citations must resolve.
    bad = sorted({c for c in CITATION_RE.findall(summary)
                  if not 1 <= int(c) <= len(event.live)})
    if bad:
        out.append(f"citation(s) {bad} point outside the {len(event.live)}-article context")

    # 6. An all-headline-only cluster must decline its implications.
    body = summary.lower()
    if NO_SUPPORT.lower() not in body:
        out.append(f"missing the required {NO_SUPPORT!r} for unsupported sections")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=TRIALS)
    ap.add_argument("--show", action="store_true", help="print each summary")
    args = ap.parse_args()

    event = build_event()
    ctx = brief._numbered_context(event)
    assert ctx.count("HEADLINE ONLY") == len(FIXTURE), "fixture must be fully headline-only"
    assert "FULL TEXT" not in ctx
    print(f"fixture ok - {len(FIXTURE)} articles, all labelled HEADLINE ONLY\n")

    if not llm.available():
        summary = brief.summarize_event(build_event(), "NVDA", use_llm=False).summary
        assert not violations(summary, event), violations(summary, event)
        print(f"llm {llm.backend()!r} unreachable - extractive fallback checked instead; "
              f"re-run with a backend up to gate the generated path")
        return 0

    failed = 0
    for trial in range(1, args.trials + 1):
        ev = brief.summarize_event(build_event(), "NVDA", use_llm=True)
        assert ev.summary_method == "llm", "expected the generated path"
        found = violations(ev.summary, event)
        if args.show:
            print(f"--- trial {trial} ---\n{ev.summary}\n")
        if found:
            failed += 1
            print(f"trial {trial}: FAIL")
            for v in found:
                print(f"    - {v}")
        else:
            print(f"trial {trial}: ok - no claim beyond the headlines")

    if failed:
        print(f"\nHEADLINE-ONLY GATE FAILED ({failed}/{args.trials} trials)")
        return 1
    print(f"\nHEADLINE-ONLY GATE PASSED ({args.trials}/{args.trials} trials)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
