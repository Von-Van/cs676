#!/usr/bin/env python3
"""
evaluate.py — score TinyTroupe (or fallback) persona conversations.

Input:  JSONL file produced by simulate.py  (records with fields in TurnRecord)
Optional: personas.json to enrich consistency checks

Output: Markdown report summarizing metrics & notes per persona.

Usage:
  python evaluate.py --in runs/smoke_run.jsonl --report runs/smoke_report.md --personas personas.json
"""

import argparse, json, math, statistics, re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Iterable, Optional

# ----------------------- Utils -----------------------

def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def load_personas(path: Optional[str]) -> Dict[str, dict]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {p["id"]: p for p in data}

_word_re = re.compile(r"[A-Za-z0-9']+")
def words(text: str) -> List[str]:
    return [w.lower() for w in _word_re.findall(text)]

def sentences(text: str) -> List[str]:
    # Simple sentence split; good enough for heuristics
    return re.split(r"(?<=[.!?])\s+", text.strip())

def type_token_ratio(tokens: List[str]) -> float:
    return (len(set(tokens)) / max(1, len(tokens)))

def question_ratio(text: str) -> float:
    sents = [s for s in sentences(text) if s]
    if not sents:
        return 0.0
    qs = sum(1 for s in sents if s.strip().endswith("?"))
    return qs / len(sents)

def repeated_bigram_rate(tokens: List[str]) -> float:
    if len(tokens) < 2:
        return 0.0
    bigrams = list(zip(tokens, tokens[1:]))
    c = Counter(bigrams)
    repeats = sum(1 for k, v in c.items() if v > 1)
    return repeats / max(1, len(c))

def keyword_hits(text: str, keywords: Iterable[str]) -> int:
    toks = set(words(text))
    return sum(1 for k in keywords if k.lower() in toks)

def flatten(lst):
    return [x for y in lst for x in y]

# ----------------------- Scoring -----------------------

def score_realism(avg_len: float, q_ratio: float, rep_rate: float) -> float:
    """
    Heuristic:
      - Avg assistant message length 35-220 words is best
      - Some questions are okay but not all (0-0.5 ideal)
      - Low repetition preferred
    """
    s_len = 1.0 - min(1.0, abs((avg_len - 100.0) / 100.0))  # peak at ~100 words
    s_q   = 1.0 - min(1.0, abs(q_ratio - 0.20) / 0.20)      # peak ~20% questions
    s_rep = 1.0 - min(1.0, rep_rate * 8.0)                  # heavy penalty for repeats
    composite = max(0.0, (0.4*s_len + 0.3*s_q + 0.3*s_rep))
    return round(composite * 5.0, 2)

def score_diversity(ttr: float, rep_rate: float) -> float:
    # High TTR & low repetition => better diversity
    s_ttr = min(1.0, ttr / 0.6)          # 0.6+ TTR is excellent for short turns
    s_rep = 1.0 - min(1.0, rep_rate*8.0)
    composite = 0.6*s_ttr + 0.4*s_rep
    return round(max(0.0, composite) * 5.0, 2)

def score_consistency(assistant_texts: List[str], persona: Optional[dict]) -> float:
    if not persona:
        # If no persona metadata provided, estimate by stability of tone (variance in length & q-ratio)
        lens = [len(words(t)) for t in assistant_texts]
        qrs  = [question_ratio(t) for t in assistant_texts]
        if len(lens) < 2:
            return 2.5
        vlen = statistics.pstdev(lens) / max(1.0, statistics.mean(lens))
        vqr  = statistics.pstdev(qrs) / max(1e-6, statistics.mean(qrs) or 1.0)
        raw  = 1.0 - min(1.0, (0.6*vlen + 0.4*vqr))
        return round(raw * 5.0, 2)

    # With persona: look for trait/goal/style keywords showing up consistently
    kw = set()
    for k in ("traits", "goals"):
        for item in persona.get(k, []):
            kw.update(words(str(item)))
    kw.update(words(str(persona.get("style", ""))))
    if "demographics" in persona:
        for v in persona["demographics"].values():
            kw.update(words(str(v)))

    if not kw:
        return 2.5

    hits_per_turn = [keyword_hits(t, kw) for t in assistant_texts]
    avg_hits = statistics.mean(hits_per_turn) if hits_per_turn else 0.0
    # Normalize: ~3+ hits per turn is strong consistency for short replies
    s_hits = min(1.0, avg_hits / 3.0)

    # Penalize wild swings in tone length
    lens = [len(words(t)) for t in assistant_texts]
    if len(lens) >= 2:
        vlen = statistics.pstdev(lens) / max(1.0, statistics.mean(lens))
        s_stab = 1.0 - min(1.0, vlen)   # lower variance => better
    else:
        s_stab = 0.6

    composite = 0.7*s_hits + 0.3*s_stab
    return round(max(0.0, composite) * 5.0, 2)

# ----------------------- Main evaluation -----------------------

def evaluate(records: List[dict], personas_by_id: Dict[str, dict]) -> Tuple[str, dict]:
    """
    Returns (markdown_report, metrics_dict)
    """
    # Group assistant turns per persona
    by_persona: Dict[str, List[str]] = defaultdict(list)
    by_persona_user: Dict[str, List[str]] = defaultdict(list)
    latencies: Dict[str, List[float]] = defaultdict(list)

    for r in records:
        pid = r.get("persona_id", "unknown")
        role = r.get("role")
        content = r.get("content", "")
        if role == "assistant":
            by_persona[pid].append(content)
            latencies[pid].append(float(r.get("latency_s", 0.0)))
        elif role == "user":
            by_persona_user[pid].append(content)

    # Compute metrics
    rows_md = []
    summary = {}
    for pid, texts in by_persona.items():
        persona_meta = personas_by_id.get(pid)
        all_tokens = flatten([words(t) for t in texts])
        avg_len = statistics.mean([len(words(t)) for t in texts]) if texts else 0.0
        q_ratio = statistics.mean([question_ratio(t) for t in texts]) if texts else 0.0
        rep_rate = repeated_bigram_rate(all_tokens) if all_tokens else 0.0
        ttr = type_token_ratio(all_tokens) if all_tokens else 0.0
        lat = statistics.mean(latencies.get(pid, [0.0]))

        realism = score_realism(avg_len, q_ratio, rep_rate)
        diversity = score_diversity(ttr, rep_rate)
        consistency = score_consistency(texts, persona_meta)

        # Simple comments
        notes = []
        if rep_rate > 0.08:
            notes.append("noticeable repetition")
        if q_ratio > 0.45:
            notes.append("asks too many questions")
        if avg_len < 25:
            notes.append("very short answers")
        if avg_len > 220:
            notes.append("very long/rambling answers")

        # Save
        summary[pid] = {
            "avg_len_words": round(avg_len, 1),
            "question_ratio": round(q_ratio, 3),
            "repeated_bigram_rate": round(rep_rate, 3),
            "type_token_ratio": round(ttr, 3),
            "avg_latency_s": round(lat, 2),
            "score_realism": realism,
            "score_consistency": consistency,
            "score_diversity": diversity,
            "notes": notes,
        }

        pname = personas_by_id.get(pid, {}).get("name", pid)
        rows_md.append(
            f"| {pname} | {avg_len:.1f} | {q_ratio:.2f} | {ttr:.2f} | {rep_rate:.3f} | {lat:.2f} | {realism:.2f} | {consistency:.2f} | {diversity:.2f} | {'; '.join(notes) or '—'} |"
        )

    # Overall section
    header = (
        "# Conversation Quality Report\n\n"
        "This report summarizes heuristic metrics for realism, persona consistency, and response diversity.\n"
        "Higher scores (0–5) are better. Heuristics are lightweight and meant for **comparative** evaluation.\n\n"
    )
    table_header = (
        "| Persona | Avg Len (w) | ?-Ratio | TTR | Repeat Rate | Avg Lat (s) | Realism (0-5) | Consistency (0-5) | Diversity (0-5) | Notes |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n"
    )
    body = "\n".join(rows_md) if rows_md else "_No assistant turns found._"

    # Simple guidance section
    guidance = (
        "\n\n## Interpretation & Next Steps\n"
        "- **Realism** favors medium-length, low-repetition answers with a natural amount of questions (~20%).\n"
        "- **Consistency** improves when trait/goal/style keywords appear regularly and answer lengths are stable.\n"
        "- **Diversity** rewards broader vocabulary (high TTR) and low phrase repetition.\n"
        "\n### Suggested follow-ups\n"
        "1. Tweak persona **style/goals** prompts where consistency is low.\n"
        "2. Add guardrails to reduce repeated phrasing (e.g., \"avoid repeating earlier wording\").\n"
        "3. If answers are too short/long, set explicit target lengths in the system prompt.\n"
    )

    md = header + table_header + body + guidance
    return md, summary

# ----------------------- CLI -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input JSONL from simulate.py")
    ap.add_argument("--report", required=True, help="Output Markdown path")
    ap.add_argument("--personas", default=None, help="Optional personas.json for consistency checks")
    args = ap.parse_args()

    records = read_jsonl(args.inp)
    personas_by_id = load_personas(args.personas)
    md, _ = evaluate(records, personas_by_id)

    with open(args.report, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Wrote report to: {args.report}")

if __name__ == "__main__":
    main()
