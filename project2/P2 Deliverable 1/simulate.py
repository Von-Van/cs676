#!/usr/bin/env python3
"""
simulate.py — run short conversations for each persona and log to JSONL.

- Reads personas from personas.json
- Uses TinyTroupe if available; otherwise falls back to OpenAI
- Saves one JSONL file with all turns: runs/<timestamp>_run.jsonl
"""

import argparse, json, os, time, uuid, sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

# --- Optional .env for API keys ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# --- TinyTroupe (preferred) or OpenAI fallback ---
USE_TINY = False
try:
    import tinytroupe  # noqa
    USE_TINY = True
except Exception:
    pass

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Fallback client only constructed if needed
def _openai_chat(messages, max_tokens=300):
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.7,
    )
    return resp.choices[0].message.content

# ---- Data classes ----
@dataclass
class Persona:
    id: str
    name: str
    demographics: Dict
    traits: List[str]
    goals: List[str]
    style: str

@dataclass
class TurnRecord:
    run_id: str
    persona_id: str
    persona_name: str
    turn_index: int
    role: str
    content: str
    latency_s: float
    timestamp: float

# ---- Utilities ----
def load_personas(path: str) -> List[Persona]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [Persona(**p) for p in raw]

def ensure_dir(p: str):
    os.makedirs(os.path.dirname(p), exist_ok=True)

# ---- Conversation runner ----
def build_system_prompt(persona: Persona) -> str:
    return (
        f"You are role-playing as a specific user persona.\n"
        f"Persona name: {persona.name}\n"
        f"Demographics: {persona.demographics}\n"
        f"Core traits: {', '.join(persona.traits)}\n"
        f"Goals: {', '.join(persona.goals)}\n"
        f"Speaking style: {persona.style}\n\n"
        f"Stay in character at all times. Be consistent across turns."
    )

# Scenarios (Step 3.2) — you can edit later
SCENARIOS = {
    "curious_student": "I’m prepping for an exam on decision trees. Explain entropy and information gain simply, then quiz me.",
    "skeptical_engineer": "We deployed a model with 91% accuracy. Convince me it’s production-ready—address data drift, monitoring, and failure modes.",
    "teen_gamer": "Ranked is rough lately. Give me 3 actionable tips to climb this week, based on common mistakes in competitive shooters.",
    "healthcare_researcher": "We’re proposing LLM-based note summarization for clinical trials. What are key ethical risks and mitigations?",
    "small_business_owner": "I run a 6-person e-commerce shop. Outline a lightweight plan to use AI for support emails with costs and timeline."
}

def initial_user_prompt(persona: Persona) -> str:
    return SCENARIOS.get(persona.id, "Start a conversation relevant to your persona and ask one clarifying question.")

def call_model(system_prompt: str, history: List[Dict], user_msg: str) -> str:
    # TinyTroupe stub — adapt if you want to use native Agents/APIs
    if USE_TINY:
        # Example shape (adjust to your TinyTroupe version):
        # agent = tinytroupe.Agent(system_prompt)
        # return agent.chat(user_msg)
        # For portability (till you wire TinyTroupe), fall back to OpenAI:
        pass
    # Fallback to OpenAI chat
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": user_msg}]
    return _openai_chat(messages)

def run_conversation(persona: Persona, turns: int) -> List[TurnRecord]:
    system_prompt = build_system_prompt(persona)
    history: List[Dict[str, str]] = []
    records: List[TurnRecord] = []
    run_id = str(uuid.uuid4())
    user_msg = initial_user_prompt(persona)

    for t in range(turns):
        # USER -> MODEL
        t0 = time.time()
        assistant = call_model(system_prompt, history, user_msg)
        latency = time.time() - t0
        # Log both user and assistant turns
        records.append(TurnRecord(run_id, persona.id, persona.name, t*2, "user", user_msg, 0.0, time.time()))
        records.append(TurnRecord(run_id, persona.id, persona.name, t*2+1, "assistant", assistant, latency, time.time()))
        # Update history and craft next turn
        history.extend([{"role": "user", "content": user_msg}, {"role": "assistant", "content": assistant}])
        # Simple next prompt: ask the persona to push conversation forward
        user_msg = "Continue the conversation in a way your persona naturally would. Ask one specific follow-up."

    return records

def write_jsonl(path: str, records: List[TurnRecord]):
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

# ---- CLI ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--personas", default="personas.json")
    ap.add_argument("--turns", type=int, default=5)
    ap.add_argument("--out", default=f"runs/{int(time.time())}_run.jsonl")
    args = ap.parse_args()

    personas = load_personas(args.personas)
    all_records: List[TurnRecord] = []
    for p in personas:
        print(f"Running persona: {p.name} ({p.id}) for {args.turns} turns...")
        recs = run_conversation(p, args.turns)
        all_records.extend(recs)

    write_jsonl(args.out, all_records)
    print(f"Saved transcripts to: {args.out}")

if __name__ == "__main__":
    if not USE_TINY and not OPENAI_API_KEY:
        print("Warning: TinyTroupe not detected and OPENAI_API_KEY not set. Set one to run conversations.", file=sys.stderr)
    main()
