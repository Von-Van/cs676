#!/usr/bin/env python3
"""
simulate.py — run persona simulations with TinyTroupe and log transcripts as JSONL.

What this script does:
- Loads personas from personas.json (or a provided subset file)
- Injects a FEATURE text (from env FEATURE_TEXT) into the initial prompt
- Generates multi-turn conversations (TinyTroupe or Mock Mode)
- Captures per-turn metadata: reasoning_summary, confidence (0–1), followups
- Writes a JSONL transcript to --out (supports --append)

Usage (CMD):
  set FEATURE_TEXT=Order-tracking widget on receipts page
  python simulate.py --turns 3 --personas personas.json --out runs\\test_run.jsonl

Optional env:
  SIM_USE_MOCK=1         # bypasses OpenAI calls and synthesizes plausible replies
  SIM_TURN_SLEEP=2.0     # seconds to sleep after each assistant turn (default 2.0)
"""

import argparse
import json
import os
import time
from uuid import uuid4
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple

# ---------------------- Config toggles ----------------------

USE_MOCK = os.getenv("SIM_USE_MOCK", "0") == "1"
TURN_SLEEP_S = float(os.getenv("SIM_TURN_SLEEP", "2.0"))

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
    turn_index: int        # 0-based across the conversation (user=even, assistant=odd)
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

def write_jsonl(path: str, turn_records: List[TurnRecord], append: bool = False):
    ensure_parent_dir(path)
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
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
        return f"The feature to evaluate: {feature_text}\n\n{base}"
    return base

# ---------------------- TinyTroupe bridge ----------------------

def _build_person(persona_name: str):
    """
    Load the example Lisa spec but auto-rename the agent before it registers,
    avoiding 'Agent name ... is already in use.'
    """
    from tinytroupe.examples import load_example_agent_specification
    from tinytroupe.agent.tiny_person import TinyPerson

    spec = load_example_agent_specification("Lisa")  # stable base spec
    # Give each instance a unique name based on the persona (prevents registry collisions)
    return TinyPerson.load_specification(
        spec,
        auto_rename_agent=True,
        new_agent_name=f"{persona_name}-{uuid4().hex[:6]}"
    )

# ---------------------- Mock mode ----------------------

def _mock_reply(persona_name: str, user_msg: str) -> Tuple[str, Dict[str, Any]]:
    """
    Deterministic, persona-flavored stub reply + metadata.
    Keeps runs fast and offline for demos/tests.
    """
    import random, hashlib
    seed = int(hashlib.sha256((persona_name + "|" + user_msg).encode()).hexdigest(), 16) % (2**32)
    random.seed(seed)

    tone = {
        "Curious Student": ("curious", ["why does this matter?", "how will it update?", "can you show an example?"]),
        "Busy Parent": ("hurried", ["is it one tap?", "can I do this from email?"]),
        "Small Business Owner": ("practical", ["what's the ROI?", "what if tracking fails?"]),
    }.get(persona_name, ("neutral", ["any edge cases?", "show an example?"]))

    style, qs = tone
    adjectives = {
        "curious": ["curious", "detail-seeking"],
        "hurried": ["time-pressed", "mobile-first"],
        "practical": ["cost-aware", "risk-focused"],
        "neutral": ["balanced"]
    }
    adj = random.choice(adjectives.get(style, ["balanced"]))

    reply = (
        f"As a {adj} {persona_name.lower()}, this seems useful if it reduces uncertainty quickly. "
        f"I'd want a visible 'last updated' timestamp and a clear fallback when tracking fails. "
        f"Could you provide one concrete example of the user flow?"
    )
    meta = {
        "confidence": round(0.58 + random.random() * 0.24, 2),
        "reasoning": f"{style} assessment focused on clarity and next action.",
        "followups": random.sample(qs, k=min(2, len(qs)))
    }
    return reply, meta

# ---------------------- Generation helpers ----------------------

def _parse_meta_from_reply(text: str) -> Tuple[str, Dict[str, Any]]:
    """
    If the reply ends with a line like:
      META: {"confidence":0.73,"reasoning":"...","followups":["q1","q2"]}
    extract it into a dict and remove the line from the visible content.
    """
    meta = {}
    lines = text.splitlines()
    if lines:
        last = lines[-1].strip()
        if last.startswith("META:"):
            payload = last.replace("META:", "", 1).strip()
            try:
                meta = json.loads(payload)
                text = "\n".join(lines[:-1]).rstrip()
            except Exception:
                meta = {}
    return text, meta

def _generate_reply_with_metadata(history: List[Dict[str, str]],
                                  system_prompt: str,
                                  user_msg: str,
                                  persona_name: str) -> Tuple[str, float, Dict[str, Any]]:
    """
    Returns (assistant_text, latency_seconds, meta_dict)
    Uses mock mode when SIM_USE_MOCK=1; otherwise calls TinyTroupe/OpenAI.
    """
    t0 = time.time()

    if USE_MOCK:
        text, meta = _mock_reply(persona_name, user_msg)
        return text, time.time() - t0, meta

    # --- Real model path (TinyTroupe)
    person = _build_person(persona_name)

    if system_prompt:
        person.listen(f"[SYSTEM] {system_prompt}")
    for h in history:
        role = h.get("role", "user").upper()
        content = h.get("content", "")
        if content.strip():
            person.listen(f"[{role}] {content}")

    # Single call: reply + META line appended
    request = (
        f"{user_msg}\n\n"
        "At the very end of your message, append exactly one line starting with:\n"
        "META: {\"confidence\":0.xx,\"reasoning\":\"<=20 words\",\"followups\":[\"q1\",\"q2\"]}\n"
        "Do not include extra JSON or additional META lines."
    )

    try:
        assistant = person.listen_and_act(request) or "(no response)"
    except KeyboardInterrupt:
        raise
    except Exception as e:
        assistant = f"(generation error: {e})"

    latency = time.time() - t0
    assistant, meta = _parse_meta_from_reply(assistant)
    return assistant, latency, meta

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

def _clamp_float(v: Any) -> Optional[float]:
    try:
        x = float(v)
        if x < 0: x = 0.0
        if x > 1: x = 1.0
        return round(x, 3)
    except Exception:
        return None

def run_conversation(persona: Persona, turns: int, feature_text: str) -> List[TurnRecord]:
    system_prompt = build_system_prompt(persona)
    history: List[Dict[str, str]] = []
    records: List[TurnRecord] = []
    run_id = str(uuid4())

    user_msg = initial_user_prompt(persona.id, feature_text)

    for t in range(turns):
        # Log user turn
        records.append(
            TurnRecord(
                run_id=run_id,
                persona_id=persona.id,
                persona_name=persona.name,
                feature_text=feature_text,
                turn_index=t * 2,
                role="user",
                content=user_msg,
                latency_s=0.0,
                timestamp=time.time(),
            )
        )

        # Assistant turn
        assistant, latency, meta = _generate_reply_with_metadata(
            history, system_prompt, user_msg, persona_name=persona.name
        )

        records.append(
            TurnRecord(
                run_id=run_id,
                persona_id=persona.id,
                persona_name=persona.name,
                feature_text=feature_text,
                turn_index=t * 2 + 1,
                role="assistant",
                content=assistant,
                latency_s=latency,
                timestamp=time.time(),
                reasoning_summary=meta.get("reasoning"),
                confidence=_clamp_float(meta.get("confidence")),
                followups=meta.get("followups") if isinstance(meta.get("followups"), list) else None,
            )
        )

        # brief pause after assistant turn to avoid hammering the API / simulate pacing
        time.sleep(TURN_SLEEP_S)

        # Update history and craft next user message
        history.extend([
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant},
        ])
        user_msg = "Continue in character. Provide one concrete improvement suggestion and ask one specific follow-up."

    return records

# ---------------------- CLI ----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--personas", default="personas.json", help="Path to personas JSON")
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--out", default=f"runs/{int(time.time())}_run.jsonl")
    ap.add_argument("--append", action="store_true", help="Append to --out instead of overwriting")
    args = ap.parse_args()

    feature_text = os.getenv("FEATURE_TEXT", "").strip()

    personas = load_personas(args.personas)
    all_records: List[TurnRecord] = []

    for p in personas:
        print(f"Running persona: {p.name} ({p.id}) for {args.turns} turns...")
        recs = run_conversation(p, args.turns, feature_text)
        all_records.extend(recs)

    write_jsonl(args.out, all_records, append=args.append)
    print(f"Saved transcripts to: {args.out}")

if __name__ == "__main__":
    main()
