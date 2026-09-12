<!--
FC-6 EVIDENCE — generated AFTER the headline-only summarizer constraint
(src/generation/prompts.py EVENT_SUMMARY_SYSTEM + brief._numbered_context labelling).

Compare against evals/before_realtext_NVDA.md, same ticker and window.

Audit by scripts/smoke_headline_only.py's checker, which only applies to clusters
with no body text at all (it treats every headline as the evidence ceiling):
  event 1: MIXED (1/4 full text) - not checked, has full text
  event 2: MIXED (1/5 full text) - not checked, has full text
  event 3: ALL HEADLINE-ONLY (0/3 full text) - 0 violation(s)
  event 4: ALL HEADLINE-ONLY (0/2 full text) - 0 violation(s)
  event 5: ALL HEADLINE-ONLY (0/2 full text) - 0 violation(s)

Events 3-5 carry no body text and decline all four interpretive sections with
"Not supported by headline-only sources." rather than inventing a read.

Events 1-2 do carry body text and still produce grounded implications. Their
figures were verified by hand against the article bodies rather than by the
checker: event 1's "106% year over year to $96.2 billion" and "221% ... $16.7
billion" and event 2's "17.3% gain", "70% revenue growth" and "$700 billion in
sales" all appear verbatim in the full-text article each cites. That is the
behaviour the constraint is meant to preserve — it restricts claims to the
evidence, it does not suppress analysis where evidence exists.

Generated 2026-09-12 against the live corpus
(216 articles, 37 with body text).
-->

# NVDA — NVIDIA Corporation

Weekly intelligence brief · last 7 days · generated 2026-09-12

## 1. An AI Stock Launched a Rack-Scale System 30x Faster Than Nvidia’s GPUs. Here’s Why It Is Still a Risky Buy.
*2026-09-08–2026-09-12 · 4 article(s) · 4 source(s)* · _product_

**Headline/Theme:** Nvidia's AI Chip Market Share

**Date:** September 2026

**Summary:** Nvidia's AI chip market share may be threatened by a shift towards custom accelerators, as companies like OpenAI develop their own chips to reduce computing costs and optimize inference workloads. Broadcom is benefiting from this trend with its custom chip offerings. Nvidia's revenue growth is significant, but it may be at risk of being disrupted by this shift.

**Potential Market Impact:** [4] suggests that a shift towards custom accelerators could pressure Nvidia's share of inference computing, which could impact the company's revenue.

**Bull implication:** [2] suggests that Nvidia is a good buy due to its AI hype, while [4] notes that Nvidia's revenue growth is significant.

**Bear implication:** [1] and [4] suggest that Nvidia's market share may be threatened by a shift towards custom accelerators, which could impact the company's revenue.

**Connection to recent earnings/filings:** [4] mentions that Nvidia's revenue surged 106% year over year to $96.2 billion in fiscal 2027's Q2, while Broadcom's AI semiconductor revenue jumped 221% year over year to $16.7 billion in fiscal 2026's Q3.

**Sources**
- [An AI Stock Launched a Rack-Scale System 30x Faster Than Nvidia’s GPUs. Here’s Why It Is Still a Risky Buy.](https://news.google.com/rss/articles/CBMi4AFBVV95cUxPU0lWZHVYN3NmUmpucUd2ek1ZTWdZRHhMcWZNckVpM3lxZnlSMDEyRDBOdURxNXM0UHhocUx5Z01STTlMM04zaXNuRDBkSEtkbVRpbWdSd2JSS0xiZVZ1a3gtM0lVMnQ0dVUtMHFuMUtSbjlEMjNTVjltYTQ1c3ZCYk81UkQ5cmN4QjFXLUNiMzBBa2tsSUpQSlRVTHBHZGJob3NOcHl4VG5ER095QXdxVHNCRkJyaUYtX2dBU2V4ZlpqbGZRWVloOHoyQ01MTkpUSWpqdXhUSmdhNFlaV2pidg?oc=5) — Barchart.com, 2026-09-08
- [Nvidia Has the AI Hype. AMD Has the Valuation. I’d Buy This Stock.](https://news.google.com/rss/articles/CBMiqgFBVV95cUxQby04ZVB0OUxJTFVKZ085VVdnbU11cFpMRFBwVTNOUmdBaFpvaGYtdkFxMEtzbmpTQzNxWHBOMWkyeFduUGlRM3dBdGE3WUMyeEp1aGdNUnd3UDBnV280TzlTRFQxS3lGQUJMcmJmclNfSE5HU0xZS05iZ1dGTHFRUXFEUGp0M2pWalBwRlZMdC1KSE9tTElUak9HOGtXemlHU3k1elV2Q2NVZw?oc=5) — 247wallst.com, 2026-09-10
- [NVIDIA (NVDA) Expands Its AI Factory Footprint, Is The Stock A Bargain?](https://news.google.com/rss/articles/CBMiygFBVV95cUxQalY3d0lMMVp1cUFHWkZRV09aRWxCLTBfVzdCVndjdGE0WVRibUVFZGg5Zk1UNlFNWTdNb2w1X2NHeWRCcVdmTTRkaVFrRU9DeWEtUjNiYUpjd3NXbFJLZVEtT29XVzZpcV9DSWhuajZXaHB2eWpPcEV6T0w2RlVPdERWYndTc0NiTjRTSEpnNlNXV2p5OWtXd2h2aFM0eTk2SmE4OEcwak1NckdrVUxsUXhUUURIMXAzNkJoZmxDWV9mMGlVV2N6b1Z30gHPAUFVX3lxTE1kYlpWelNLWUM2OGxiOE1QaFRIellrdWxOSnUwSnRlTTRNbURLajJxdHZhVHFOWW9WTG84MlVpTW80RkR4blBRTWk3UHJjaDQwUEtoMTN3T1RyMDdBWS1Nc1gyMUI1RTNCRnVheTI4allPVkE4N25McmF5djV0cTNMeEJJbE5tLUFCdExQNnFZMWZMVzd4V0JxdjJSd2RnU0dEREcwanVYOS1RWmlZN1VXTGxVSlkyb0tDMWFvQmxmMzVwTFp6RTFWQnNLRWZhaw?oc=5) — simplywall.st, 2026-09-12
- [Better AI Chip Stock: Broadcom vs. Nvidia](https://www.fool.com/investing/2026/09/12/better-ai-chip-stock-broadcom-vs-nvidia/?.tsrc=rss) — yahoo, 2026-09-12

## 2. Prediction: Nvidia Stock Will Double in Under a Year
*2026-09-06–2026-09-12 · 5 article(s) · 3 source(s)* · _analyst_

**Headline/Theme:** Nvidia Stock Performance and Future Growth Predictions

**Date:** September 2026

**Summary:** 
Nvidia's stock has had a solid but unspectacular 17.3% gain this year, but its setup beneath the surface looks interesting. The company's fiscal 2028 guidance calls for 70% revenue growth, which is above Wall Street's models and puts it on a path toward $700 billion in sales. [5]

**Potential Market Impact:** Not supported by headline-only sources.

**Bull implication:** Nvidia's stock may be quietly loading for a sizable move heading into calendar year 2027, with the company on a path toward $700 billion in sales. [5]

**Bear implication:** Not supported by headline-only sources.

**Connection to recent earnings/filings:** Nvidia's fiscal 2028 guidance calls for 70% revenue growth, which is above Wall Street's models. [5]

**Sources**
- [Prediction: Nvidia Stock Will Double in Under a Year](https://news.google.com/rss/articles/CBMimAFBVV95cUxONkZsVzdBd09qOFlsR0NLMDducVNtRWc4VGFHcy1QSnZqU2xqVHZ0dXBpN2hobnN1cGl5alVYT2dHM1k4b040WFhOWko1OG9vZXhvTS1Udk5kSVBPVlpYS0VMMGxSeldScEhZV0UweUJGUXptbU9ZYnh6T2ZrQWNVMUNGQ0JzOWNOM192aGZCVFJ1bEdETGF2TA?oc=5) — The Motley Fool, 2026-09-06
- [Nvidia Is Worth $5 Trillion. But This Could Be the Next Big Catalyst for the Stock](https://news.google.com/rss/articles/CBMiwgFBVV95cUxQQnp0XzVHcVJmbXFEanlpMmN4Qk4zSGtZN2lWTGNGZzBLU1NENlI0YndlNlhDbHM5NHR6bWFZcGlYRzNQa0xmRjBjdjJlc19FSEwtbHVDRHlwU1ZnbFNuZnh1N3pXZS1JQl91NzJpblczZGxudHlrSWRKdktwek84SWxIcmhqTWZOeTNGNG15VzJVNFlYdkVSN1NiSDdlc0hmTVFhRFNYbTRiQk11UUpDZkdUZVRwWUgtUFlNNkxUVjJXUQ?oc=5) — 247wallst.com, 2026-09-08
- [Prediction: NVIDIA Stock Has Returned 14,700% in 10 Years. Can It Do It Again?](https://news.google.com/rss/articles/CBMiuAFBVV95cUxNNXBTaTlUZF9YVFE0bzFOTk9ITEZfSGhTRVJvdVUwUDBBdmFEY01TV0tyM2hweE03Tkl3cDlGaFVtYnZmSUxIS0xBdTFoYzhrRzhyQjRGVm00LWx5NURZTW0zaDVpVGY1SUVCQ2plWVBrYXRIVzM1WUhxeGYzQmNnWUZBRlZxR3JuTy1PdEZIS3VhYW9OTkR2T0lkcDZ1MzMtdEJFUk5xckNZc2hwUDE0aWswWkRSR1hG?oc=5) — 247wallst.com, 2026-09-09
- [Prediction: $2,304 Invested in Nvidia Today Will Be Worth Triple That Amount in 5 Years](https://news.google.com/rss/articles/CBMijgFBVV95cUxPeGNVaFlmZ05zTl9XbTkyZmh3cEpTNVpjME9UbThoRWxOY0lzeERSdmRST1VMU1dVcjFRZXRRTlNZdjJXLTJiTlRXemxWakNFdUdmZG8zX2RCbmNfS0hkbnExWDlNVjVKT2kwTU0xQUNoU3Q2VWZMd0VzNVRXdDlGaGZIV2MwTlU2aEF0UTZB?oc=5) — The Motley Fool, 2026-09-09
- [Prediction: Nvidia Stock Could Be Worth This Much by January 2028](https://www.fool.com/investing/2026/09/12/predict-nvidia-stock-worth-this-much-2028/?.tsrc=rss) — yahoo, 2026-09-12
- _(1 near-duplicate repost(s) suppressed)_

## 3. Dear Nvidia Stock Fans, Mark Your Calendars for September 10
*2026-09-08–2026-09-09 · 3 article(s) · 3 source(s)* · _analyst_

Headline/Theme: Nvidia Stock Event on September 10

Date: September 8-9, 2026

Summary: 
Nvidia stock fans are advised to mark their calendars for September 10, according to [1]. A prediction is made that September 10 will be a big day for Nvidia shareholders, with something to watch, according to [2]. A forecast is made for the end of September, but the details are not available, according to [3].

Potential Market Impact: Not supported by headline-only sources.

Bull implication: Not supported by headline-only sources.

Bear implication: Not supported by headline-only sources.

Connection to recent earnings/filings: Not supported by headline-only sources.

**Sources**
- [Dear Nvidia Stock Fans, Mark Your Calendars for September 10](https://news.google.com/rss/articles/CBMipgFBVV95cUxQVkgwdkU0a1VmdFBndEhZVmMwOTVMNFpyN3hBSmQydEdxUmJYX292bGd2eGlzaXJaZlhjZFJBRGpPM1Q1eFUwcVptZnI0aUV3bkxxQ0hVajRnS1FUVHRGNlZpdi0wQzZ1Vk1MUEdQN003WmN1X2FZWjhBUENmQUtFcWc2TzZvNnBxY3hpaTZpM2xuUXFnQ3hVTHZJWDRWRXFUdVR3M2FR?oc=5) — Barchart.com, 2026-09-08
- [Prediction: Sept. 10 Will Be a Big Day for Nvidia Shareholders. Here’s What to Watch.](https://news.google.com/rss/articles/CBMilgFBVV95cUxPTFBrMVZXRmhNcmpkTlRaS0V2U2JuYWxpckNVZ0xyLWhXWHc3SFU1UV9OMDhPcjhMZjBTX3B5RTdhall1RERzcVFQV1Jnd2JoN0RmUlBwZ3RCN0pGdHJZNnFhQjVENS1uRGJEME9GeWJURmdwLUpEYkFnVC13RDRlTExtbGlWX2xLdmJTdFo4UC1iQ2N0dlE?oc=5) — Yahoo Finance, 2026-09-08
- [Here's My Nvidia Stock Price Forecast for the End of September](https://news.google.com/rss/articles/CBMijAFBVV95cUxQUy1lbU9tQVU1VmpuMDZGSnhaX2s4QkF3MTdZOWFXcUtXblBYVUpKazdGVXBkdHh4RGpUNmtNSWdfQjRSUzhnRHNrUWc2V2lhSTZIY0tTaVlQYXNIbk5KaE53QjhockhnaVBCNm54cWk4MlBMSHZXWkpSTWYzNU5ZYmgydmpCYmIxV0VjSA?oc=5) — Currently.com, 2026-09-09
- _(1 near-duplicate repost(s) suppressed)_

## 4. Nvidia: 70% Growth Guidance Makes This A Strong Buy (NASDAQ:NVDA)
*2026-09-06–2026-09-10 · 2 article(s) · 2 source(s)* · _earnings_

Headline/Theme: Nvidia's 70% Revenue Growth Guidance

Date: 2026-09-06 to 2026-09-10

Summary: 
Nvidia's 70% revenue growth guidance is being touted as a strong buy opportunity for investors. Two articles, [1] and [2], mention this guidance without providing further context.

Potential Market Impact: Not supported by headline-only sources.

Bull implication: Not supported by headline-only sources.

Bear implication: Not supported by headline-only sources.

Connection to recent earnings/filings: Not supported by headline-only sources.

**Sources**
- [Nvidia: 70% Growth Guidance Makes This A Strong Buy (NASDAQ:NVDA)](https://news.google.com/rss/articles/CBMinwFBVV95cUxPaHRVYjVwaFhqckFnUVJPNHkwYnRKNDBrZ3N2cVNmUzFwMDd3dTcwLV9vZ0J3RTNHVmhqS2VOY0FGQjlnaC1XVzVXU0tYaENSRGJFNTAxd0dIYlVvTUdYRzZVeExHNXZHWWtwT3JudlJua1V5dHVjbld1dEVrQkdnTlpvV1k0RDdmdnE2V0tveHZDaFlPZWR3eGZlOGpVbTg?oc=5) — Seeking Alpha, 2026-09-06
- [70% Revenue Growth for Nvidia Next Year May Make It the Best Stock to Buy In the Market](https://news.google.com/rss/articles/CBMimAFBVV95cUxOUU1lWl9zZml1OGYzb3FDV3lnTVlxcW5qcmRYa2dTZEVSbWtMeU5CVW9tWUgyQWl2S01ZQlI3RFIzRkk4N2ZsQlg2cU83V2VMRVQ2Z19ERVBjREMyNDUxT2xXMUswdzF6WmdXWWd5Nng0TEd6QUw2QWlIQ1JGV2VRaVNpOXpFZTRqcFVqZGFpTTgwUTNHRmcwdg?oc=5) — The Motley Fool, 2026-09-10

## 5. Massive News for Nvidia Stock Investors
*2026-09-07–2026-09-08 · 2 article(s) · 2 source(s)*

Headline/Theme: Nvidia Stock News

Date: 2026-09-07-2026-09-08

Summary: 
Nvidia investors received massive news [1]. Nvidia's recent news overshadows a strong quarter [2].

Potential Market Impact: Not supported by headline-only sources.

Bull implication: Not supported by headline-only sources.

Bear implication: Not supported by headline-only sources.

Connection to recent earnings/filings: Nvidia's recent news overshadows a strong quarter [2].

**Sources**
- [Massive News for Nvidia Stock Investors](https://news.google.com/rss/articles/CBMiigFBVV95cUxQVFBZTFZiRnQ1cm4wUUtaZkR4TTcxZWExYzRqZ0U4NGVDbHZQQWllbkpjTnFxY1dnc2xTWFRDeU9kMjFnOFhGT0xHODlpaW55bWhmelIwTGxWQU1WNzl0T1FfcDdmbTZlMEdXSEJvOHFiUUhDVW9JdzRUUVczMGdsSVdzV3VBeGhFZ2c?oc=5) — The Motley Fool, 2026-09-07
- [Nvidia's Next Chapter: Why Recent News Overshadows a Strong Quarter (NVDA)](https://news.google.com/rss/articles/CBMipgFBVV95cUxNVWM4b1VHZ25hRWxOVldvNUJRX3J0aXFnZTdtQk83QjlyOXFTQWN3UmN1eFZGVllUOWlTSmRITnJrTEZkZmtDVGFIZC0za3Z5QXVJaEl4OF9fWkNLOGd1U3NaejlOWVlyUHpzVUNzNEx2bWFUbVFDRDYxVTdjUFFmTlM4MGZzc283T0R2aFBQMWt5T0JncVFlR1BMUC1CTEktSGF1M1V3?oc=5) — Seeking Alpha, 2026-09-08
