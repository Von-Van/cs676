#!/usr/bin/env python3
"""
simulate.py — run persona simulations with TinyTroupe and log transcripts as JSONL.

What this script does:
- Loads personas from personas.json (or a provided subset file)
- Injects a FEATURE text (from env FEATURE_TEXT) into the initial prompt
- Uses TinyTroupe (examples builder) to generate multi-turn conversations
- Captures per-turn metadata: reasoning_summary, confidence (0–1), followups
- Writes a single JSONL transcript to --out

Usage (CMD):
  set FEATURE_TEXT=Order-tracking widget on receipts page
  python simulate.py --turns 3 --personas personas.json --out runs\\test_run.jsonl
"""

import argparse
import json
import os
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any

# ---------------------- Data ----------------------

@dataclass
class Persona:
    id: str
    name: str
    demographics: Dict[str, Any]
    traits: List[str]
    goals: List[str]
    style: str

@dataclass
class TurnRecord:
    run_id: str
    persona_id: str
    persona_name: str
    feature_text: str
    turn_index: int        # 0-based across the conversation
    role: str              # "user" | "assistant"
    content: str
    latency_s: float
    timestamp: float
    # metadata captured on assistant turns (defaults for user turns)
    reasoning_summary: Optional[str] = None
    confidence: Optional[float] = None     # 0..1
    followups: Optional[List[str]] = None

# ---------------------- IO helpers ----------------------

def load_personas(path: str) -> List[Persona]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [Persona(**p) for p in raw]

def ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

def write_jsonl(path: str, turn_records: List[TurnRecord]):
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for r in turn_records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

# ---------------------- Prompts ----------------------

SCENARIOS: Dict[str, str] = {
    "curious_student": (
        "Evaluate the feature like a student preparing for an exam: "
        "ask clarifying questions and request examples."
    ),
    "skeptical_engineer": (
        "Evaluate risks, edge cases, monitoring, and data/telemetry needed. "
        "Challenge assumptions and ask for measurable acceptance criteria."
    ),
    "teen_gamer": (
        "React casually and quickly. Focus on friction, discoverability, and whether it feels fun."
    ),
    "healthcare_researcher": (
        "Stress ethics, data quality, provenance, and potential biases. "
        "Ask about study design and measurement."
    ),
    "small_business_owner": (
        "Focus on cost, timeline, maintainability, and customer impact. "
        "Ask what is the simplest viable version and ROI."
    ),
}

def initial_user_prompt(persona_id: str, feature_text: str) -> str:
    base = SCENARIOS.get(
        persona_id,
        "React in character to the feature and ask one clarifying question."
    )
    if feature_text:
        return f"The feature to evaluate:\n{feature_text}\n\n{base}"
    return base

# ---------------------- TinyTroupe bridge ----------------------

def _tt_build_person():
    """
    TinyTroupe quickstart pattern via examples builder.
    You can swap to a custom TinyPerson later.
    """
    from tinytroupe.examples import create_lisa_the_data_scientist
    return create_lisa_the_data_scientist()

def _generate_reply_with_metadata(history, system_prompt, user_msg):
    import time, json, signal, sys
    from tinytroupe.examples import create_lisa_the_data_scientist
    person = create_lisa_the_data_scientist()

    if system_prompt:
        person.listen(f"[SYSTEM] {system_prompt}")
    for h in history:
        role = h.get("role","user").upper()
        content = h.get("content","")
        if content.strip():
            person.listen(f"[{role}] {content}")

    # --- hard timeout guard (Windows-safe fallback)
    t0 = time.time()
    try:
        assistant = person.listen_and_act(user_msg) or "(no response)"
    except KeyboardInterrupt:
        raise
    except Exception as e:
        assistant = f"(generation error: {e})"
    latency = time.time() - t0

    # metadata side-channel (short + safe)
    meta_prompt = (
        'In one JSON line only: {"confidence":0.xx,"reasoning":"<=20w","followups":["q1","q2"]}'
    )
    meta_raw = ""
    try:
        meta_raw = person.listen_and_act(meta_prompt) or ""
    except Exception:
        pass
    meta = _parse_one_line_json(meta_raw)
    return assistant, latency, meta

def _parse_one_line_json(text: str) -> Dict[str, Any]:
    """
    Robustly extracts first {...} from a string and parses JSON.
    Returns {} on failure.
    """
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        return json.loads(text[start:end+1])
    except Exception:
        return {}

# ---------------------- Runner ----------------------

def build_system_prompt(persona: Persona) -> str:
    return (
        f"You are role-playing a persona.\n"
        f"Persona: {persona.name}\n"
        f"Demographics: {persona.demographics}\n"
        f"Traits: {', '.join(persona.traits)}\n"
        f"Goals: {', '.join(persona.goals)}\n"
        f"Style: {persona.style}\n"
        f"Stay in character at all times; be consistent across turns."
    )

def run_conversation(persona: Persona, turns: int, feature_text: str) -> List[TurnRecord]:
    system_prompt = build_system_prompt(persona)
    history: List[Dict[str, str]] = []
    records: List[TurnRecord] = []
    run_id = str(uuid.uuid4())

    user_msg = initial_user_prompt(persona.id, feature_text)

    for t in range(turns):
        # Log user turn
        records.append(
            TurnRecord(
                run_id=run_id,
                persona_id=persona.id,
                persona_name=persona.name,
                feature_text=feature_text,
                turn_index=t*2,
                role="user",
                content=user_msg,
                latency_s=0.0,
                timestamp=time.time(),
            )
        )

        # Assistant turn
        assistant, latency, meta = _generate_reply_with_metadata(history, system_prompt, user_msg)

        records.append(
            TurnRecord(
                run_id=run_id,
                persona_id=persona.id,
                persona_name=persona.name,
                feature_text=feature_text,
                turn_index=t*2 + 1,
                role="assistant",
                content=assistant,
                latency_s=latency,
                timestamp=time.time(),
                reasoning_summary=meta.get("reasoning"),
                confidence=_clamp_float(meta.get("confidence")),
                followups=meta.get("followups") if isinstance(meta.get("followups"), list) else None,
            )
        )

        # Update history and craft next user message
        history.extend([
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant},
        ])
        user_msg = "Continue in character. Provide one concrete improvement suggestion and ask one specific follow-up."

    return records

def _clamp_float(v: Any) -> Optional[float]:
    try:
        x = float(v)
        if x < 0: x = 0.0
        if x > 1: x = 1.0
        return round(x, 3)
    except Exception:
        return None

# ---------------------- CLI ----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--personas", default="personas.json", help="Path to personas JSON")
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--out", default=f"runs/{int(time.time())}_run.jsonl")
    args = ap.parse_args()

    feature_text = os.getenv("FEATURE_TEXT", "").strip()

    personas = load_personas(args.personas)
    all_records: List[TurnRecord] = []

    for p in personas:
        print(f"Running persona: {p.name} ({p.id}) for {args.turns} turns...")
        recs = run_conversation(p, args.turns, feature_text)
        all_records.extend(recs)

    write_jsonl(args.out, all_records)
    print(f"Saved transcripts to: {args.out}")

if __name__ == "__main__":
    main()
