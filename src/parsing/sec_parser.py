"""Parse raw SEC filing HTML into (section, text) tuples.

Strategy: strip HTML to plain text (one line per block element), then locate
Item-header boundaries with a line-anchored regex. 10-Ks/10-Qs open with a
table of contents that repeats every Item heading near the top of the
document — cheaply distinguished from the real section because a TOC entry
is followed almost immediately by the next Item heading, while a real
section body runs for thousands of characters before the next one. For each
wanted item id we keep the occurrence with the largest gap to the next Item
heading of any kind. 8-Ks don't have this Item-as-section structure (their
Item numbers denote event types, not report sections), so they're returned
as a single "body" section — the EX-99 press release text is what matters.

10-Qs reuse item numbers across Part I ("FINANCIAL INFORMATION") and Part II
("OTHER INFORMATION") with unrelated meanings — Part I Item 2 is MD&A, but
Part II Item 2 is "Unregistered Sales of Equity Securities". Matching on
item id alone mislabels one as the other, so wanted 10-Q items also record
the Part they must fall under; 10-K item numbers aren't reused across parts
so no Part scoping is needed there.

Known limitation (see DECISIONS.md / evals/failure_cases.md): some filers
(observed: JPM) never restate the "Item 1." header before the real Financial
Statements content, only in the table of contents — the span heuristic then
has no real occurrence to prefer and under-sizes that section. NVDA and MSFT
10-Qs restate every header and parse cleanly. Acceptable for v1; would need
an HTML-structure-aware pass (e.g. detect page-number-suffixed TOC lines) to
fix generally.
"""
import re
import warnings

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# item_id -> (section_name, required_part | None for "any part")
SECTIONS_10K = {
    "1": ("Business", None),
    "1A": ("Risk Factors", None),
    "7": ("MD&A", None),
    "8": ("Financial Statements", None),
}
SECTIONS_10Q = {
    "1": ("Financial Statements", "I"),
    "2": ("MD&A", "I"),
    "1A": ("Risk Factors", "II"),
}

ITEM_RE = re.compile(
    r'^item\s+(1a|1b|1c|7a|9a|9b|10|11|12|13|14|15|[1-9])\.?\s',
    re.IGNORECASE | re.MULTILINE,
)
PART_RE = re.compile(r'^part\s+(iv|iii|ii|i)\b', re.IGNORECASE | re.MULTILINE)


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def extract_sections(html: str, form: str) -> list[tuple[str, str]]:
    """Returns [(section_name, text), ...] for the item headers we care about.
    Falls back to a single ("body", text) section for forms without a
    tracked Item map (8-K) or when no header matches."""
    wanted = SECTIONS_10K if form == "10-K" else SECTIONS_10Q if form == "10-Q" else None
    text = html_to_text(html)
    if not text:
        return []
    if wanted is None:
        return [("body", text)]

    part_starts = sorted((m.start(), m.group(1).upper()) for m in PART_RE.finditer(text))

    def part_at(pos: int) -> str | None:
        cur = None
        for p_pos, p_id in part_starts:
            if p_pos > pos:
                break
            cur = p_id
        return cur

    all_starts = [m.start() for m in ITEM_RE.finditer(text)]
    candidates = [(m.start(), m.group(1).upper()) for m in ITEM_RE.finditer(text)
                  if m.group(1).upper() in wanted]
    candidates = [
        (pos, item_id) for pos, item_id in candidates
        if wanted[item_id][1] is None or part_at(pos) == wanted[item_id][1]
    ]
    if not candidates:
        return [("body", text)]

    best_start: dict[str, tuple[int, int]] = {}
    for pos, item_id in candidates:
        next_pos = next((p for p in all_starts if p > pos), len(text))
        span = next_pos - pos
        if item_id not in best_start or span > best_start[item_id][1]:
            best_start[item_id] = (pos, span)

    boundaries = sorted(((pos, item_id) for item_id, (pos, _) in best_start.items()))

    sections = []
    for i, (start, item_id) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((wanted[item_id][0], body))
    return sections
