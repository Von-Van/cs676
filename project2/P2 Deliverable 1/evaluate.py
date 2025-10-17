#!/usr/bin/env python3
"""
evaluate.py — score persona conversations from simulate.py JSONL.

Adds:
- Average assistant confidence (0..1)
- Keeps Realism / Consistency / Diversity heuristics
- Notes section

Usage:
  python evaluate.py --in runs\\test_run.jsonl --report runs\\test_report.md --personas personas.json
"""

import argparse, json, statistics, re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Iterable, Optional

# ----------------------- IO -----------------------

def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def load_personas(path: Optional[str]) -> Dict[str, dict]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {p["id"]: p for p in data}

# ----------------------- Text helpers -----------------------

_word_re = re.compile(r"[A-Za-z0-9']+")

def words(text: str) -> List[str]:
    return [w.lower() for w in _word_re.findall(text)]

def sentences(text: str) -> List[str]:
    return re.split(r"(?<=[.!?])\s+", text.strip())

def question_ratio(text: str) -> float:
    sents = [s for s in sentences(text) if s]
    if not sents:
        return 0.0
    qs = sum(1 for s in sents if s.strip().endswith("?"))
    return qs / len(sents)

def repeated_bigram_rate(tokens: List[str]) -> float:
    if len(tokens) < 2: return 0.0
    bigrams = list(zip(tokens, tokens[1:]))
    c = Counter(bigrams)
    repeats = sum(1 for _, v in c.items() if v > 1)
    return repeats / max(1, len(c))

def type_token_ratio(tokens: List[str]) -> float:
    return len(set(tokens)) / max(1, len(tokens))

def flatten(lst):
    return [x for y in lst for x in y]

# ----------------------- Scoring -----------------------

def score_realism(avg_len: float, q_ratio: float, rep_rate: float) -> float:
    s_len = 1.0 - min(1.0, abs((avg_len - 100.0) / 100.0))
    s_q   = 1.0 - min(1.0, abs(q_ratio - 0.20) / 0.20)
    s_rep = 1.0 - min(1.0, rep_rate * 8.0)
    composite = max(0.0, (0.4*s_len + 0.3*s_q + 0.3*s_rep))
    return round(composite * 5.0, 2)

def score_diversity(ttr: float, rep_rate: float) -> float:
    s_ttr = min(1.0, ttr / 0.6)
    s_rep = 1.0 - min(1.0, rep_rate*8.0)
    composite = 0.6*s_ttr + 0.4*s_rep
    return round(max(0.0, composite) * 5.0, 2)

def score_consistency(assistant_texts: List[str], persona: Optional[dict]) -> float:
    import statistics
    if not assistant_texts:
        return 0.0

    if not persona:
        lens = [len(words(t)) for t in assistant_texts]
        qrs  = [question_ratio(t) for t in assistant_texts]
        if len(lens) < 2:
            return 2.5
        vlen = statistics.pstdev(lens) / max(1.0, statistics.mean(lens))
        vqr  = statistics.pstdev(qrs) / max(1e-6, statistics.mean(qrs) or 1.0)
        raw  = 1.0 - min(1.0, (0.6*vlen + 0.4*vqr))
        return round(raw * 5.0, 2)

    # With persona metadata: keywords presence + stability
    kw = set()
    for k in ("traits", "goals"):
        for item in persona.get(k, []):
            kw.update(words(str(item)))
    kw.update(words(str(persona.get("style", ""))))
    for v in persona.get("demographics", {}).values():
        kw.update(words(str(v)))

    hits_per_turn = [sum(1 for w in set(words(t)) if w in kw) for t in assistant_texts]
    avg_hits = statistics.mean(hits_per_turn) if hits_per_turn else 0.0
    s_hits = min(1.0, avg_hits / 3.0)

    lens = [len(words(t)) for t in assistant_texts]
    if len(lens) >= 2:
        vlen = statistics.pstdev(lens) / max(1.0, statistics.mean(lens))
        s_stab = 1.0 - min(1.0, vlen)
    else:
        s_stab = 0.6

    composite = 0.7*s_hits + 0.3*s_stab
    return round(max(0.0, composite) * 5.0, 2)

# ----------------------- Evaluation -----------------------

def evaluate(records: List[dict], personas_by_id: Dict[str, dict]) -> Tuple[str, dict]:
    by_persona_asst: Dict[str, List[str]] = defaultdict(list)
    latencies: Dict[str, List[float]] = defaultdict(list)
    confidences: Dict[str, List[float]] = defaultdict(list)

    for r in records:
        if r.get("role") == "assistant":
            pid = r.get("persona_id", "unknown")
            by_persona_asst[pid].append(r.get("content", ""))
            latencies[pid].append(float(r.get("latency_s", 0.0)))
            conf = r.get("confidence", None)
            if isinstance(conf, (int, float)):
                confidences[pid].append(float(conf))

    rows_md = []
    summary = {}
    for pid, texts in by_persona_asst.items():
        persona_meta = personas_by_id.get(pid)
        toks = flatten([words(t) for t in texts])
        avg_len = statistics.mean([len(words(t)) for t in texts]) if texts else 0.0
        q_ratio = statistics.mean([question_ratio(t) for t in texts]) if texts else 0.0
        rep_rate = repeated_bigram_rate(toks) if toks else 0.0
        ttr = type_token_ratio(toks) if toks else 0.0
        lat = statistics.mean(latencies.get(pid, [0.0]))
        conf = statistics.mean(confidences.get(pid, [])) if confidences.get(pid) else None

        realism = score_realism(avg_len, q_ratio, rep_rate)
        diversity = score_diversity(ttr, rep_rate)
        consistency = score_consistency(texts, persona_meta)

        notes = []
        if rep_rate > 0.08: notes.append("noticeable repetition")
        if q_ratio > 0.45:  notes.append("asks too many questions")
        if avg_len < 25:    notes.append("very short answers")
        if avg_len > 220:   notes.append("very long/rambling answers")

        summary[pid] = {
            "avg_len_words": round(avg_len, 1),
            "question_ratio": round(q_ratio, 3),
            "repeated_bigram_rate": round(rep_rate, 3),
            "type_token_ratio": round(ttr, 3),
            "avg_latency_s": round(lat, 2),
            "avg_confidence": round(conf, 3) if conf is not None else None,
            "score_realism": realism,
            "score_consistency": consistency,
            "score_diversity": diversity,
            "notes": notes,
        }
        pname = personas_by_id.get(pid, {}).get("name", pid)
        conf_s = f"{summary[pid]['avg_confidence']:.2f}" if conf is not None else "—"
        rows_md.append(
            f"| {pname} | {avg_len:.1f} | {q_ratio:.2f} | {ttr:.2f} | {rep_rate:.3f} | {lat:.2f} | {conf_s} | {realism:.2f} | {consistency:.2f} | {diversity:.2f} | {'; '.join(notes) or '—'} |"
        )

    header = (
        "# Conversation Quality Report\n\n"
        "Heuristic metrics for realism, persona consistency, and response diversity.\n"
        "Confidence is the assistant's self-reported 0–1 score (averaged).\n\n"
    )
    table_header = (
        "| Persona | Avg Len (w) | ?-Ratio | TTR | Repeat Rate | Avg Lat (s) | Avg Conf | Realism | Consistency | Diversity | Notes |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n"
    )
    body = "\n".join(rows_md) if rows_md else "_No assistant turns found._"

    guidance = (
        "\n\n## Interpretation & Next Steps\n"
        "- Realism favors medium-length, low-repetition answers with ~20% questions.\n"
        "- Consistency improves when trait/goal/style terms are present and answer lengths are stable.\n"
        "- Diversity rewards wider vocabulary and low repeated phrasing.\n"
    )

    md = header + table_header + body + guidance
    return md, summary

# ----------------------- CLI -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--personas", default=None)
    args = ap.parse_args()

    records = read_jsonl(args.inp)
    personas_by_id = load_personas(args.personas)
    md, _ = evaluate(records, personas_by_id)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Wrote report to: {args.report}")

if __name__ == "__main__":
    main()
