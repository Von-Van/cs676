#!/usr/bin/env python3
"""
export_conversation_history.py — produce a readable, annotated Markdown transcript.

- Groups by persona
- Shows the FEATURE at the top
- Displays assistant metadata (reasoning, confidence, followups) inline

Usage:
  python export_conversation_history.py --in runs\\test_run.jsonl --out conversation_history.md --personas personas.json
"""

import argparse, json, datetime
from collections import defaultdict
from typing import Dict, List, Optional

def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def load_personas(path: Optional[str]) -> Dict[str, dict]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {p["id"]: p for p in data}

def persona_header(meta: dict, pid: str) -> str:
    name = meta.get("name", pid)
    demo = meta.get("demographics", {})
    traits = ", ".join(meta.get("traits", [])) or "—"
    goals = ", ".join(meta.get("goals", [])) or "—"
    style = meta.get("style", "—")
    return (
        f"## {name} (`{pid}`)\n\n"
        f"**Demographics:** {demo}\n\n"
        f"**Traits:** {traits}\n\n"
        f"**Goals:** {goals}\n\n"
        f"**Style notes:** {style}\n"
    )

def format_turn(t: dict) -> str:
    role = t.get("role", "").capitalize()
    content = (t.get("content") or "").strip()
    lines = [f"**{role}**"]
    # blockquote content
    for ln in content.splitlines():
        if ln.strip():
            lines.append("> " + ln)
    # assistant metadata
    if role == "Assistant":
        rs = t.get("reasoning_summary")
        conf = t.get("confidence")
        fus = t.get("followups") or []
        meta_bits = []
        if rs:   meta_bits.append(f"Reasoning: {rs}")
        if conf is not None: meta_bits.append(f"Confidence: {conf:.2f}")
        if fus:  meta_bits.append("Follow-ups: " + "; ".join(str(x) for x in fus))
        if meta_bits:
            lines.append(f"> _Meta_: " + " | ".join(meta_bits))
    return "\n".join(lines) + "\n"

def export_markdown(records: List[dict], personas_by_id: Dict[str, dict]) -> str:
    by_pid: Dict[str, List[dict]] = defaultdict(list)
    features: Dict[str, str] = {}

    for r in records:
        pid = r.get("persona_id", "unknown")
        by_pid[pid].append(r)
        # remember feature (same for all turns in run/persona)
        ftxt = r.get("feature_text")
        if ftxt and pid not in features:
            features[pid] = ftxt

    for pid in by_pid:
        by_pid[pid].sort(key=lambda x: x.get("turn_index", 0))

    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    out = [
        f"# Conversation History",
        f"_Generated: {date}_",
        "",
        "Organized by persona; includes assistant metadata (reasoning, confidence, followups).",
        ""
    ]

    for pid, turns in by_pid.items():
        meta = personas_by_id.get(pid, {"name": pid})
        out.append(persona_header(meta, pid))
        feat = features.get(pid, "").strip()
        if feat:
            out.append(f"**Feature under review:** {feat}\n")
        out.append("### Transcript\n")
        for t in turns:
            out.append(format_turn(t))
        out.append("\n---\n")

    return "\n".join(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--personas", default=None)
    args = ap.parse_args()

    records = read_jsonl(args.inp)
    personas_by_id = load_personas(args.personas)
    md = export_markdown(records, personas_by_id)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Wrote conversation history to: {args.out}")

if __name__ == "__main__":
    main()
