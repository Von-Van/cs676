#!/usr/bin/env python3
"""
export_conversation_history.py
Create an annotated Markdown transcript from simulate.py JSONL output.

Usage:
  python export_conversation_history.py --in runs/test_run.jsonl --out conversation_history.md --personas personas.json
"""

import argparse, json, datetime
from collections import defaultdict
from typing import Dict, List, Optional

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

def format_turn(role: str, content: str) -> str:
    tag = "**User**" if role == "user" else "**Assistant**"
    # indent block for readability
    lines = content.strip().splitlines()
    indented = "\n".join("> " + ln for ln in lines if ln.strip() != "")
    return f"{tag}\n{indented}\n"

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
        f"**Style notes:** {style}\n\n"
    )

def auto_annotation(example_user_msgs: List[str], example_assistant_msgs: List[str]) -> str:
    # lightweight hints to seed your comments
    u0 = (example_user_msgs[0][:160] + "…") if example_user_msgs else "—"
    a0 = (example_assistant_msgs[0][:160] + "…") if example_assistant_msgs else "—"
    return (
        "### Annotations (draft)\n"
        "- **Context of first exchange:** " + u0 + "\n"
        "- **Assistant opening tone:** " + a0 + "\n"
        "- **Consistency check:** Does tone/voice stay aligned with persona across turns?\n"
        "- **Depth check:** Are answers specific, avoiding generic filler?\n"
        "- **Edge cases:** Any hallucinations, repetition, or policy issues?\n\n"
    )

def export_markdown(records: List[dict], personas_by_id: Dict[str, dict]) -> str:
    # group records by persona, then order by turn_index
    by_pid: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        by_pid[r.get("persona_id", "unknown")].append(r)
    for pid in by_pid:
        by_pid[pid].sort(key=lambda x: x.get("turn_index", 0))

    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    md = [
        f"# Conversation History",
        f"_Generated: {date}_",
        "",
        "This file organizes conversation logs **by persona** and includes space for annotations explaining key exchanges.",
        ""
    ]

    for pid, turns in by_pid.items():
        meta = personas_by_id.get(pid, {"name": pid})
        md.append(persona_header(meta, pid))

        # brief auto-annotations
        user_msgs = [t["content"] for t in turns if t.get("role") == "user"]
        asst_msgs = [t["content"] for t in turns if t.get("role") == "assistant"]
        md.append(auto_annotation(user_msgs, asst_msgs))

        md.append("### Transcript\n")
        for t in turns:
            md.append(format_turn(t.get("role", ""), t.get("content", "")))
        md.append("\n---\n")

    return "\n".join(md)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input JSONL from simulate.py")
    ap.add_argument("--out", required=True, help="Output Markdown path")
    ap.add_argument("--personas", default=None, help="Optional personas.json for headers")
    args = ap.parse_args()

    records = read_jsonl(args.inp)
    personas_by_id = load_personas(args.personas)
    md = export_markdown(records, personas_by_id)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Wrote conversation history to: {args.out}")

if __name__ == "__main__":
    main()
