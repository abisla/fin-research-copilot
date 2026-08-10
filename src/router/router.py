"""Query router: rules first, LLM fallback.

Routes: NEWS | FILING_RAG | NUMERIC | MIXED | CROSS_SOURCE
Rules catch the cheap, unambiguous cases (fast, free, deterministic, testable).
LLM fallback handles genuinely ambiguous phrasing.
Interview q: "why not LLM for everything?" -> latency + cost + non-determinism
in eval; rules give you a testable contract for the 80% case.
TODO(Phase 6): implement classify(query) -> Route, log to routing_log table.
"""
import re

NUMERIC_PAT = re.compile(r"\b(revenue|eps|margin|operating income|growth|guidance number|how much|what was)\b", re.I)
NEWS_PAT = re.compile(r"\b(news|last week|this week|recent(ly)? (announced|reported)|past \d+ days|headlines?)\b", re.I)
FILING_PAT = re.compile(r"\b(10-?[kq]|8-?k|risk factors?|md&a|management (said|discuss)|earnings call|commentary)\b", re.I)
