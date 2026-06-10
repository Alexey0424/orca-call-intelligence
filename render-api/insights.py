"""Headline / verdict / recommendations / highlights / coaching score.

All of these are deterministic functions of the flat KPIs — no LLM calls,
no surprises. They live in one module so the templates and the executive
view share the same wording.

**Highlights are money/decision-focused.** What survives onto the report's
top strip is what a sales manager wants on the first glance:
  • the client's stated price ask (e.g. "Client asking $65k")
  • whether a next-step was secured (and what)
  • the highest-signal objection (verbatim, with timestamp)
  • the highest-signal buying signal (verbatim, with timestamp)
We extract dollar amounts from the LLM event quotes with a regex; if
nothing matches we fall back to a sentiment-extreme utterance.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import config
from utils import fnum, format_duration, trim


_REC_LIMIT     = 5
_HIGHLIGHTS_K  = 4
_QUOTE_MAX_LEN = 130


# ── Headline (one short sentence; replaces the old multi-clause text) ───────
def headline(*, next_step: bool, objections: int,
             buying_signals: int, risk_level: str,
             client_sentiment: Dict[str, Any]) -> str:
    trajectory = str(client_sentiment.get("sentiment_trajectory", "stable")).lower()

    if next_step and buying_signals > 0:
        return ("Strong close: forward momentum, buyer engagement, and a "
                "documented next step.")
    if not next_step and objections >= 2 and trajectory == "declining":
        return ("Friction call: unresolved objections, declining sentiment, "
                "and no clear next step.")
    if not next_step:
        return ("Engaged conversation, but no confirmed next step — "
                "follow-through is at risk.")
    if risk_level == "low":
        return ("Balanced and operationally sound; minimal friction "
                "across the call.")
    return ("Generally balanced, with coaching opportunities around "
            "objection-handling and close clarity.")


# ── Verdict line (the hero pull-quote, one sentence in quotation marks) ─────
def verdict(*, next_step: bool, objections: int, buying_signals: int,
            interruptions: int, rapport: Optional[str],
            outcome: Optional[str],
            client_sentiment: Dict[str, Any]) -> str:
    """One narrative sentence to anchor the hero block."""
    trajectory = str(client_sentiment.get("sentiment_trajectory", "stable")).lower()
    rapport = (rapport or "").lower()

    if next_step and buying_signals > 0 and rapport != "low":
        return ("This was a strong, well-paced call: rapport held, the "
                "client surfaced buying signals, and the next step is "
                "documented.")
    if not next_step and objections >= 2 and trajectory == "declining":
        return ("The call went sideways: objections compounded, client "
                "sentiment slipped, and the conversation ended without "
                "a path forward.")
    if not next_step and buying_signals > 0:
        return ("The buyer signalled interest, but the call ended without "
                "a confirmed next step — the most concrete coaching "
                "opportunity here is the close.")
    if not next_step:
        return ("Engagement was acceptable, but the absence of a confirmed "
                "next step puts follow-through at risk.")
    if interruptions >= config.INTERRUPTIONS_HEAVY:
        return ("The mechanics held — there is a next step — but heavy "
                "interruptions suggest the conversation needs cleaner "
                "turn-taking.")
    if rapport == "high":
        return ("A composed, balanced call. Rapport was strong; routine "
                "coaching applies.")
    return ("A balanced call overall, with room to sharpen objection "
            "handling and tighten the close.")


# ── Coaching score (deterministic 0..100, with audit breakdown) ─────────────
_GRADE_TABLE: List[Tuple[int, str]] = [
    (85, "A"), (75, "B"), (65, "C"), (50, "D"), (0, "F"),
]


def coaching_score(*, next_step: bool, objections: int, interruptions: int,
                   silence_ratio: float, buying_signals: int,
                   rapport: Optional[str],
                   client_sentiment: Dict[str, Any]
                   ) -> Tuple[int, str, List[Dict[str, Any]]]:
    """Return ``(score, grade, breakdown)``.

    Breakdown is a list of ``{factor, delta}`` dicts so the score can be
    audited / explained. ``score`` is clamped to [0, 100].
    """
    trajectory = str(client_sentiment.get("sentiment_trajectory", "stable")).lower()
    rapport_lvl = (rapport or "").lower()

    breakdown: List[Dict[str, Any]] = [{"factor": "base", "delta": 100}]

    def add(factor: str, delta: int) -> None:
        if delta:
            breakdown.append({"factor": factor, "delta": delta})

    if not next_step:                                   add("no_next_step",        -15)
    if objections >= config.OBJECTIONS_HEAVY:           add("objection_count_high",-10)
    if interruptions >= config.INTERRUPTIONS_HEAVY:     add("interruptions_high",   -8)
    if silence_ratio >= config.SILENCE_HEAVY_RATIO:     add("silence_ratio_high",   -6)
    if trajectory == "declining":                       add("client_sent_declining",-10)
    if buying_signals == 0:                             add("no_buying_signals",    -6)
    if rapport_lvl == "low":                            add("rapport_low",         -12)
    if rapport_lvl == "high":                           add("rapport_high",          3)
    if next_step and buying_signals > 0:                add("close_alignment",       5)

    raw = sum(b["delta"] for b in breakdown)
    score = max(0, min(100, raw))

    grade = next(g for cut, g in _GRADE_TABLE if score >= cut)
    return score, grade, breakdown


# ── Recommendations (chip-style, with priority + headline + rationale) ──────
def recommendations(*, next_step: bool, objections: int, interruptions: int,
                    silence_ratio: float, buying_signals: int,
                    client_sentiment: Dict[str, Any],
                    closer_text: Dict[str, Any]) -> List[Dict[str, str]]:
    """Up to ``_REC_LIMIT`` recommendations as ``{priority, headline, rationale}``."""
    out: List[Dict[str, str]] = []

    def push(headline: str, rationale: str) -> bool:
        out.append({
            "priority": "high" if len(out) < 2 else "medium" if len(out) < 4 else "low",
            "headline": headline,
            "rationale": rationale,
        })
        return len(out) >= _REC_LIMIT

    if not next_step:
        if push("Lock the next step before hanging up.",
                "Close every call with a concrete next step, owner, and "
                "date. Implied follow-up is the single biggest source of "
                "lost deals."):
            return out
    if objections >= config.OBJECTIONS_HEAVY:
        if push("Tighten the objection sequence.",
                "Acknowledge → isolate the concern → respond directly → "
                "confirm resolution. This call had multiple objections "
                "without confirmation steps."):
            return out
    if interruptions >= config.INTERRUPTIONS_HEAVY:
        if push("Coach for cleaner turn-taking.",
                f"{interruptions} interruptions detected. Heavy "
                "overlap leaves clients feeling unheard even when the "
                "content is strong."):
            return out
    if silence_ratio >= config.SILENCE_HEAVY_RATIO:
        if push("Reduce dead-air with intentional transitions.",
                f"{silence_ratio*100:.0f}% of the call was silence. "
                "Use confirmation prompts and bridge phrases to keep "
                "momentum."):
            return out
    if str(client_sentiment.get("sentiment_trajectory", "")).lower() == "declining":
        if push("Pivot earlier when sentiment slips.",
                "Client sentiment trended downward. Watch for the inflection "
                "and pivot to reassurance or evidence before resistance "
                "compounds."):
            return out
    if buying_signals == 0:
        if push("Surface intent with discovery questions.",
                "No buying signals detected. Probe urgency, budget, and "
                "timeline explicitly — implicit interest is often illusion."):
            return out
    if int(fnum(closer_text.get("question_count"), 0)) < 3:
        push("Lead with more discovery questions.",
             f"Closer asked only "
             f"{int(fnum(closer_text.get('question_count'), 0))} questions. "
             "Discovery-led calls let the closer guide rather than react.")

    if not out:
        out.append({
            "priority": "low",
            "headline": "Maintain current structure.",
            "rationale": "No major risk patterns detected. Keep documenting "
                         "the next step on every call.",
        })
    return out


# ── Money / decision extraction ─────────────────────────────────────────────
# Match $65k, $65,000, 65,000, 65k, sixty-five thousand, etc.
# Also: domain-aware bare numbers (e.g. "I'll take 65" → $65k) when preceded by
# a price verb in a real-estate conversation. Plausible price range 30-999.
_PRICE_PATTERNS = [
    re.compile(r"\$\s?(\d{1,3}(?:,\d{3})+)(?:\.\d+)?\b"),                      # $65,000
    re.compile(r"\$\s?(\d+(?:\.\d+)?)\s?([kKmM])\b"),                          # $65k, $1.2M
    re.compile(r"\b(\d{2,3}(?:,\d{3})+)(?:\.\d+)?\s*(?:k|K|grand|thousand|usd|dollars?)?\b"),
    re.compile(r"\b(\d{2,4})\s?([kKmM])\b"),                                   # 65k, 1m
    re.compile(r"\b(\d{2,4})\s+(?:thousand|grand)\b", re.IGNORECASE),         # 65 thousand
]
# Domain heuristic: in a real-estate call, "take/want/asking/for/at/do/accept N"
# where N ∈ [30, 999] is almost always a price in thousands. False positives
# (e.g. "do 65 days") are extremely rare in this corpus.
_VERB_PRICE_RE = re.compile(
    r"\b(?:take|takes|taking|want|wants|wanted|asking|ask|for|at|do|accept|"
    r"accepted|sell|selling|buy|buying|pay|paid|offer|offered|need)\s+"
    r"(\d{2,3})\b",
    re.IGNORECASE,
)


def _extract_price(text: str) -> Optional[str]:
    """Return the most prominent dollar amount found in ``text``, normalised
    to a human form (``$65k``, ``$650,000``, ``$1.2M``). Returns None if no
    plausible price is present.
    """
    if not text:
        return None
    t = text.strip()

    # $65,000 / 65,000
    for pat in (_PRICE_PATTERNS[0], _PRICE_PATTERNS[2]):
        m = pat.search(t)
        if m:
            raw = m.group(1).replace(",", "")
            try:
                n = int(raw)
            except ValueError:
                continue
            if n >= 1000:
                return f"${n:,}"

    # $65k / $1.2M / 65k / 1m
    for pat in (_PRICE_PATTERNS[1], _PRICE_PATTERNS[3]):
        m = pat.search(t)
        if m:
            try:
                base = float(m.group(1))
            except ValueError:
                continue
            unit = m.group(2).lower()
            if unit == "k":
                return f"${base:g}k"
            if unit == "m":
                return f"${base:g}M"

    # "65 thousand"
    m = _PRICE_PATTERNS[4].search(t)
    if m:
        try:
            n = int(m.group(1))
            return f"${n}k"
        except ValueError:
            pass

    # Domain-aware bare-number: "I'll take 65" → $65k
    m = _VERB_PRICE_RE.search(t)
    if m:
        try:
            n = int(m.group(1))
            if 30 <= n <= 999:
                return f"${n}k"
        except ValueError:
            pass

    return None


def _scan_events_for_price(events: Dict[str, Any]
                            ) -> Optional[Dict[str, Any]]:
    """Scan objection + buying-signal events for the first dollar amount and
    return a highlight-shaped dict, or None."""
    for ev in (events.get("objection_events") or []
               ) + (events.get("buying_signal_events") or []):
        text = (ev.get("text") or "").strip()
        price = _extract_price(text)
        if price:
            return {
                "kind":       "price_point",
                "time_sec":   round(fnum(ev.get("start_sec", ev.get("start")), 0), 2),
                "time_human": format_duration(fnum(ev.get("start_sec", ev.get("start")), 0)),
                "role":       "client",
                "text":       f"Client asking {price} — \"{trim(text, 90)}\"",
                "icon":       "$",
            }
    return None


# ── Event filtering — keep substance, drop short acks ───────────────────────
_MIN_EVENT_WORDS = 4           # require at least N words to count as a real event
# Phrases that often appear as the LLM's "buying_signal" or "objection" but are
# really just acknowledgements / fillers. Lowercase, exact-or-startswith match.
_ACK_PATTERNS = (
    "yeah", "yep", "yes", "okay", "ok", "alright", "all right",
    "mm-hmm", "uh-huh", "sure", "right", "got it", "gotcha",
    "i see", "okay okay",
)


def _is_substantive(text: str) -> bool:
    """Return True if ``text`` looks like a real event quote (not an ack).

    Two-step filter:
      1. Hard minimum word count (``_MIN_EVENT_WORDS`` = 4).
      2. **If the utterance starts with an ack ("Yeah,", "Okay", "Right",
         …) we require a higher threshold (10+ tokens) — the ack prefix
         steals signal, so short "Yeah, so that'll be right. Okay."-style
         filler-sandwiches get rejected even when they're 6-7 words long.
    """
    t = (text or "").strip().lower()
    if not t:
        return False

    tokens = t.split()
    if len(tokens) < _MIN_EVENT_WORDS:
        return False

    # Detect leading ack
    head = t.rstrip(".,!?…").strip()
    starts_with_ack = False
    for ack in _ACK_PATTERNS:
        if head == ack or head.startswith(ack + " ") or head.startswith(ack + ","):
            starts_with_ack = True
            break

    # Detect trailing ack (e.g. "…that'll be right. Okay.")
    tail_word = tokens[-1].rstrip(".,!?…").lower()
    ends_with_ack = tail_word in {a.split()[0] for a in _ACK_PATTERNS}

    if starts_with_ack and len(tokens) < 10:
        return False
    if starts_with_ack and ends_with_ack:
        # Ack sandwich — almost never substantive regardless of length
        return False
    return True


# Refusal markers — when a "buying signal" starts with these, the LLM mislabeled
# a counter-offer / objection as a buying signal. We reject these from the
# buying-signal slot specifically; the same text often still belongs in the
# objection list (LLMs frequently double-list them).
_REFUSAL_STARTS = (
    "no,", "no.", "no ",
    "not ", "nope", "nah",
    "don't ", "dont ", "do not ",
    "won't ", "wont ", "will not ",
    "can't ", "cant ", "cannot ",
    "i'll hang", "ill hang", "i will hang",
    "i'll pass", "ill pass",
    "i'll keep", "ill keep",
)


def _is_real_buying_signal(text: str) -> bool:
    """Stricter than ``_is_substantive``: also rejects refusal-flavoured text.

    The LLM frequently puts the *client's price ask* into ``buying_signal_events``
    even when it's framed as a refusal ("No, I'll take 65" — i.e., I will NOT
    accept your offer, I want $65k). For the cover-strip highlight we want
    real green-light intent, not ambiguity. The same quote often appears in
    ``objection_events`` already, so dropping it from buying signals doesn't
    lose information — it's just relabeling.
    """
    t = (text or "").strip().lower()
    if not _is_substantive(t):
        return False
    for r in _REFUSAL_STARTS:
        if t.startswith(r):
            return False
    return True


def _first_substantive(events_list: List[Dict[str, Any]],
                       limit: int = 6,
                       predicate=None) -> Optional[Dict[str, Any]]:
    """Return the first event whose text passes ``predicate`` (default substantive)."""
    check = predicate or (lambda txt: _is_substantive(txt))
    for evt in (events_list or [])[:limit]:
        if check(evt.get("text", "")):
            return evt
    return None


# ── Highlights (top moments — money/decision/quote focused) ─────────────────
def highlights(events: Dict[str, Any],
               utterances: List[Dict[str, Any]],
               *,
               next_step: bool = False,
               next_step_text: Optional[str] = None,
               ) -> List[Dict[str, Any]]:
    """Pick up to 4 high-signal moments for the cover strip.

    Priority order:
      1. **Price point** — first dollar amount found in any event.
      2. **Next step** — secured (with text) or not secured (warning).
      3. **Top substantive buying signal** — short acks filtered out.
      4. **Top substantive objection** — short acks filtered out.

    Each item: ``{kind, time_sec, time_human, role, text, icon}``.
    """
    out: List[Dict[str, Any]] = []

    def push(kind: str, time_sec: Optional[float], role: str,
             text: str, icon: str) -> None:
        if not text:
            return
        if (time_sec is not None
                and any(h["time_sec"] is not None
                        and abs(h["time_sec"] - time_sec) < 0.5 for h in out)):
            return
        out.append({
            "kind":       kind,
            "time_sec":   round(time_sec, 2) if time_sec is not None else None,
            "time_human": format_duration(time_sec) if time_sec is not None else "",
            "role":       role,
            "text":       trim(text, _QUOTE_MAX_LEN),
            "icon":       icon,
        })

    # 1. Price point from any event
    price_hl = _scan_events_for_price(events)
    if price_hl:
        out.append(price_hl)

    # 2. Next-step status (always surface — this is the #1 manager question)
    if next_step:
        push("next_step_secured", None, "closer",
             f"Follow-up secured · {trim(next_step_text, 90)}"
                 if next_step_text else "Follow-up secured.",
             "→")
    else:
        push("next_step_missing", None, "closer",
             "No follow-up secured.", "⚠")

    # 3. Top **real** buying signal — refusal-flavoured ones get dropped
    buy = _first_substantive(events.get("buying_signal_events"),
                              predicate=_is_real_buying_signal)
    if buy:
        push("buying_signal",
             fnum(buy.get("start_sec", buy.get("start")), 0),
             "client",
             str(buy.get("text", "")),
             "✓")

    # 4. Top substantive objection
    obj = _first_substantive(events.get("objection_events"))
    if obj:
        push("objection",
             fnum(obj.get("start_sec", obj.get("start")), 0),
             "client",
             str(obj.get("text", "")),
             "⚠")

    # Backfill with the most "extreme" utterance if we somehow have <2 items
    if len(out) < 2 and utterances:
        scored = sorted(
            (s for s in utterances
                if _is_substantive(s.get("text", ""))),
            key=lambda s: abs(fnum(s.get("sentiment_score"), 0)),
            reverse=True,
        )
        for utt in scored:
            if len(out) >= _HIGHLIGHTS_K:
                break
            role = "closer" if utt.get("role") == "closer" else "client"
            push("pivot",
                 fnum(utt.get("start"), 0),
                 role,
                 str(utt.get("text", "")),
                 "★")

    return out[:_HIGHLIGHTS_K]


# ── Pull quote (the single most impactful sentence — sits before transcript) ─
def pull_quote(events: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Pick the longest *substantive* objection or buying-signal as the hero quote.

    "Substantive" rejects short acks like "Yeah, okay." even when the LLM
    classified them as buying signals or objections. We pick from objections
    first (they're typically more useful for coaching), then buying signals.
    """
    def score(evt: Dict[str, Any]) -> int:
        text = (evt.get("text") or "").strip()
        return len(text.split()) if _is_substantive(text) else 0

    candidates: List[Tuple[str, Dict[str, Any]]] = (
        [("objection",     e) for e in (events.get("objection_events") or [])] +
        [("buying_signal", e) for e in (events.get("buying_signal_events") or [])]
    )
    candidates = [(k, e) for k, e in candidates if score(e) >= _MIN_EVENT_WORDS]
    if not candidates:
        return None
    candidates.sort(key=lambda ke: score(ke[1]), reverse=True)
    kind, evt = candidates[0]
    return {
        "kind":       kind,
        "time_sec":   round(fnum(evt.get("start_sec", evt.get("start")), 0), 2),
        "time_human": format_duration(fnum(evt.get("start_sec", evt.get("start")), 0)),
        "text":       (evt.get("text") or "").strip(),
        "icon":       "✓" if kind == "buying_signal" else "⚠",
    }
