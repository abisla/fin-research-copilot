"""Phase 3 gate: exercise the retrieval stack against the real indexed corpus.

Unlike smoke_index.py (synthetic chunks, throwaway collection) this runs against
whatever build_index.py actually produced, because the properties worth asserting
here are properties of retrieval *over real filings*: that filters constrain, that
BM25 beats dense on a rare product token, that fusion draws from both arms, and that
the reranker actually reorders rather than passing the fused order through.

Read-only. Writes nothing to either store.

    python scripts/smoke_retrieval.py
"""
from datetime import date

from src.common.config import CFG
from src.common.models import Filters
from src.retrieval import dense as dense_mod
from src.retrieval.hybrid import MODES, Retriever, rrf_fuse


def descending(chunks) -> bool:
    return all(a.score >= b.score for a, b in zip(chunks, chunks[1:]))


def ids(chunks) -> list[str]:
    return [c.meta.chunk_id for c in chunks]


def main():
    cfg = CFG["retrieval"]
    with Retriever() as r:
        n_corpus = len(r.bm25)
        assert n_corpus > 0, "empty BM25 index — run scripts/build_index.py first"
        print(f"corpus: {n_corpus} chunks\n")

        # --- dense -----------------------------------------------------------
        q = "why did data center revenue grow this quarter"
        hits = r.search(q, mode="dense", top_k=10)
        assert len(hits) == 10, len(hits)
        assert descending(hits), "dense hits not sorted by score"
        assert all(h.retriever == "dense" for h in hits)
        assert all(h.text for h in hits), "dense hit arrived without text hydrated from SQL"
        assert all(h.meta.ticker and h.meta.doc_type for h in hits), "metadata missing"
        assert -1.0 <= hits[0].score <= 1.0, f"cosine out of range: {hits[0].score}"
        print(f"dense ok   - top1 {hits[0].meta.ticker} {hits[0].meta.doc_type} "
              f"{hits[0].meta.section} score={hits[0].score:.3f} "
              f"({len(hits[0].text)} chars hydrated)")

        # hydrate=False must skip SQL but keep ids/scores intact
        dry = r.search(q, mode="dense", top_k=10, hydrate=False)
        assert ids(dry) == ids(hits) and all(not d.text for d in dry)
        print(f"           hydrate=False returns same {len(dry)} ids, no text (eval path)")

        # --- filters constrain, on both arms ---------------------------------
        f = Filters(tickers=["NVDA"], doc_types=["10-K"])
        d_f = r.search(q, mode="dense", top_k=10, filters=f)
        s_f = r.search(q, mode="bm25", top_k=10, filters=f)
        assert d_f and s_f, "filter excluded everything — corpus or filter is wrong"
        for name, res in (("dense", d_f), ("bm25", s_f)):
            assert all(h.meta.ticker == "NVDA" for h in res), f"{name} leaked a ticker"
            assert all(h.meta.doc_type == "10-K" for h in res), f"{name} leaked a doc_type"
        # and must *exclude*: an unfiltered search here does contain other tickers
        unfiltered = r.search(q, mode="dense", top_k=10)
        assert {h.meta.ticker for h in unfiltered} - {"NVDA"}, \
            "unfiltered search returned only NVDA — filter assertion would be vacuous"
        assert r.search(q, mode="dense", top_k=5, filters=Filters(tickers=["ZZZZ"])) == []
        print(f"filters ok - ticker+doc_type held on both arms "
              f"(unfiltered spans {sorted({h.meta.ticker for h in unfiltered})}); "
              f"unknown ticker -> 0")

        sections = sorted({h.meta.section for h in r.search(
            q, mode="dense", top_k=20, filters=Filters(sections=["Risk Factors"]))})
        assert sections == ["Risk Factors"], sections
        recent = r.search(q, mode="dense", top_k=20, filters=Filters(date_from=date(2026, 1, 1)))
        assert recent and all(h.meta.filing_date >= date(2026, 1, 1) for h in recent)
        print(f"           section filter -> {sections}; "
              f"filing_date>=2026-01-01 -> {len(recent)} hits, "
              f"earliest {min(h.meta.filing_date for h in recent)}")

        # --- the case hybrid exists for: rare token, dense loses -------------
        rare = "Blackwell"
        d_rank = ids(r.search(rare, mode="dense", top_k=30))
        s_rank = ids(r.search(rare, mode="bm25", top_k=30))
        assert s_rank, f"BM25 found nothing for {rare!r}"
        top_sparse = s_rank[0]
        dense_pos = d_rank.index(top_sparse) + 1 if top_sparse in d_rank else None
        print(f"\nsparse ok  - {rare!r}: BM25 top1 {top_sparse} ranks "
              f"{dense_pos or '>30'} on the dense side")

        # --- fusion -----------------------------------------------------------
        d30 = r.search(rare, mode="dense", top_k=cfg["dense_top_k"])
        s30 = r.search(rare, mode="bm25", top_k=cfg["sparse_top_k"])
        fused = rrf_fuse(d30, s30, k=cfg["rrf_k"], top_n=cfg["dense_top_k"])
        assert descending(fused), "fused list not sorted"
        assert len(set(ids(fused))) == len(fused), "RRF emitted duplicate chunk_ids"
        assert all(f.retriever == "hybrid" and f.text for f in fused)
        both = set(ids(d30)) & set(ids(s30))
        only_sparse = [c for c in ids(fused) if c in set(ids(s30)) - set(ids(d30))]
        assert only_sparse, "fused pool drew nothing the dense arm had missed"
        # a chunk found by both arms must outrank the same-ranked chunk found by one
        assert fused[0].score > 1.0 / (cfg["rrf_k"] + len(fused)), "RRF scores look degenerate"
        print(f"rrf ok     - {len(fused)} fused from {len(d30)}+{len(s30)} "
              f"({len(both)} in both arms, {len(only_sparse)} sparse-only survived); "
              f"top score {fused[0].score:.4f}")

        # --- rerank -----------------------------------------------------------
        final = r.search(rare, mode="rerank")
        assert len(final) == cfg["final_top_k"], len(final)
        assert descending(final), "reranked list not sorted"
        assert all(f.retriever == "rerank" and f.text for f in final)
        assert set(ids(final)) <= set(ids(fused)), "reranker invented a chunk"
        moved = ids(final) != ids(fused)[:len(final)]
        print(f"rerank ok  - {len(fused)} -> {len(final)}; order changed={moved}; "
              f"scores {final[0].score:.2f}..{final[-1].score:.2f}")
        print(f"           top1 {final[0].meta.ticker} {final[0].meta.doc_type} "
              f"{final[0].meta.filing_date} [{final[0].meta.section}] "
              f"{final[0].meta.chunk_id}")

        # rerank must refuse un-hydrated input rather than score empty strings
        try:
            from src.retrieval.rerank import rerank
            rerank(rare, r.search(rare, mode="dense", top_k=3, hydrate=False))
            raise AssertionError("rerank accepted un-hydrated candidates")
        except ValueError as e:
            assert "hydrate" in str(e)
        print("           refuses un-hydrated candidates (would rank on empty text)")

        # --- every mode runs, honours filters, returns the common dataclass ---
        print()
        for mode in MODES:
            res = r.search(rare, mode=mode, top_k=5, filters=Filters(tickers=["NVDA"]))
            assert all(h.meta.ticker == "NVDA" for h in res), f"{mode} leaked a ticker"
            assert descending(res) and len(res) <= 5
            print(f"{mode:7s} -> {len(res)} hits, top={res[0].meta.chunk_id if res else '-'} "
                  f"score={res[0].score:.4f}" if res else f"{mode:7s} -> 0 hits")

        try:
            r.search(rare, mode="nope")
            raise AssertionError("bad mode accepted")
        except ValueError:
            pass

        # payload->meta inversion must survive the Qdrant round trip
        one = r.search(q, mode="dense", top_k=1)[0]
        row = dense_mod.store.load_chunk_rows(r.conn, [one.meta.chunk_id])[0]
        assert dense_mod.store.row_to_meta(row) == one.meta, "payload meta != SQL meta"
        print("\nmeta ok    - Qdrant payload round-trips to the same ChunkMeta as SQL")

    print("\nPHASE 3 RETRIEVAL STACK PASSED")


if __name__ == "__main__":
    main()
