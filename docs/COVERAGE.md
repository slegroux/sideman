# Coverage: how it is measured

The README says Sideman reaches the whole Live Object Model. This page is the
evidence behind that sentence, and the method for re-measuring it after a Live
release.

## The census

`scripts/census.py` asks the running Live, through the Remote Script, for every
type Ableton's own Max-for-Live machinery registers (`_MxDCore.LomTypes`) plus
the reachable types it does not register (Browser, BrowserItem, routing
objects), and for every member on each. The result is pinned to the Live
version that produced it:

```
baseline/lom_census_<version>.json
```

Live 12.2.7 exposes **47 reachable types / 922 substantive members**. The
census records which private getter spelling bound (`mxd_api`), because
Ableton has renamed those getters once inside the 12.x line already.

Re-run it after upgrading Live. The README's and landing page's numbers are
asserted against the newest census by `scripts/gen_docs.py --check`, which the
test suite and CI run, so a stale census fails the build rather than the docs
quietly drifting.

## The superset claim

`scripts/coverage_harness.py` reads the source of the three Remote Script
servers below from `~/Projects/vendor/`, extracts every Live API attribute
they touch, and checks each one against the census. It exits nonzero if any
attribute is unaccounted for, and exit 2 if a competitor checkout is missing
rather than reporting a superset it never measured.

Every attribute falls into one of four buckets, and the harness prints them:

- in the census — covered;
- competitor server infrastructure (sockets, threads, stdlib) — not Live API;
- verified absent from Live 12.2.7 — members those servers reference that Live
  12 removed, or legacy / Max-for-Live-only spellings; each was probed by
  direct `get`/`call` against a running Live with positive controls passing;
- unexplained — anything else, which is the failing case.

Currently 0 unexplained.

## Measured tool counts

Actual wired registrations, not README claims. Measured **2026-08-01** against
the vendor checkouts of that date; competitors ship, so re-measure before
citing.

| Server | Tools | Bridge |
|---|---|---|
| [jpoindexter/ableton-mcp](https://github.com/jpoindexter/ableton-mcp) | 128 | Remote Script |
| [uisato/ableton-mcp-extended](https://github.com/uisato/ableton-mcp-extended) | 46 | Remote Script |
| [xiaolaa2/ableton-copilot-mcp](https://github.com/xiaolaa2/ableton-copilot-mcp) | 42 | ableton-js |
| [ahujasid/ableton-mcp](https://github.com/ahujasid/ableton-mcp) | ~21 | Remote Script |
| [Simon-Kansara/ableton-live-mcp-server](https://github.com/Simon-Kansara/ableton-live-mcp-server) | 1 | AbletonOSC |

The harness verifies the three Remote Script servers only. xiaolaa2
(ableton-js) and Simon-Kansara (OSC) ride different bridges and are out of
scope for that method. Max-for-Live servers such as
[producer-pal](https://github.com/adamjmurray/producer-pal) and servers on
Ableton's Extensions SDK are not in this table: they reach Live through APIs
that are subsets of the Remote Script surface in the places that matter here
(the M4L LOM has no clip automation envelopes; the Extensions SDK has no
transport, browser, automation or listeners), so a tool count would not be
comparing like with like.

## Reproducing

```bash
./scripts/census.py                 # writes baseline/lom_census_<version>.json
./scripts/coverage_harness.py       # superset verdict, four buckets
./scripts/gen_docs.py --check       # README / landing page numbers vs census
```
