import os
import json
import time
import random
import configparser
from pathlib import Path
from datetime import datetime

# TinyTroupe
from tinytroupe.examples import create_lisa_the_data_scientist
from tinytroupe.worlds import TinyWorld

# --------------------
# Paths & config
# --------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
CFG_PATH = ROOT / "config" / "config.ini"

OUT_DIR.mkdir(parents=True, exist_ok=True)

CFG = configparser.ConfigParser()
if CFG_PATH.exists():
    CFG.read(CFG_PATH)
else:
    raise FileNotFoundError(f"Missing config at {CFG_PATH}")

API_KEY = os.getenv("OPENAI_API_KEY") or CFG.get("openai", "api_key", fallback="")
if not API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY (env) or put it under [openai] api_key in config/config.ini")

MODEL = CFG.get("model", "name", fallback="gpt-4.1-mini")
TEMPERATURE = CFG.getfloat("run", "temperature", fallback=0.7)
MAX_TURNS = CFG.getint("run", "max_turns", fallback=6)
SEED = CFG.get("run", "seed", fallback=(
    "Debate whether to ship an auto-context feature that summarizes the last 3 minutes "
    "of a livestream for late-joining viewers."
))

PERSONAS_PATH = DATA_DIR / "personas.json"
if not PERSONAS_PATH.exists():
    raise FileNotFoundError(f"Missing personas file at {PERSONAS_PATH}")

LOG_PATH = OUT_DIR / "conversation_log.jsonl"
README_PATH = OUT_DIR / "conversation_readme.md"

# --------------------
# Helpers
# --------------------
def backoff_sleep(attempt: int, base: float = 0.8) -> None:
    # jittered exponential backoff
    delay = base * (2 ** attempt) + random.uniform(0, 0.25)
    time.sleep(delay)

def make_agent_from_template(name, role, backstory, goals, style):
    """
    Build a TinyTroupe agent using the example DS persona as a template,
    then override persona fields to match our JSON.
    """
    agent = create_lisa_the_data_scientist(model_name=MODEL, temperature=TEMPERATURE)
    agent.persona.name = name
    agent.persona.role = role
    agent.persona.backstory = backstory
    agent.persona.goals = goals
    agent.persona.style = style
    return agent

def load_personas(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("personas.json must be a non-empty JSON array of persona objects.")
    agents = []
    for p in data:
        agents.append(
            make_agent_from_template(
                name=p.get("name", "Agent"),
                role=p.get("role", "Contributor"),
                backstory=p.get("backstory", ""),
                goals=p.get("goals", ""),
                style=p.get("style", "")
            )
        )
    return agents

# --------------------
# Run simulation
# --------------------
def run_simulation(seed: str, turns: int):
    agents = load_personas(PERSONAS_PATH)
    world = TinyWorld(name="DiscussionRoom", residents=agents)

    transcript = []
    world.broadcast(seed)

    # Round-robin: each agent speaks once per turn
    for turn in range(1, turns + 1):
        for agent in world.residents:
            # retry on transient rate limits
            last_err = None
            for attempt in range(6):
                try:
                    msg = agent.listen_and_act(
                        "Continue the group discussion with one concise, concrete point and an implicit handoff."
                    )
                    break
                except Exception as e:
                    s = str(e).lower()
                    if "429" in s or "rate" in s or "temporar" in s:
                        last_err = e
                        backoff_sleep(attempt)
                        continue
                    raise
            else:
                # exhausted retries
                raise RuntimeError(f"Backoff exhausted on model call: {last_err}") from last_err

            text = msg if isinstance(msg, str) else getattr(msg, "text", str(msg))
            transcript.append({
                "turn": turn,
                "speaker": agent.persona.name,
                "role": agent.persona.role,
                "text": (text or "").strip()
            })

    return agents, transcript

def write_jsonl(rows, path: Path):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_markdown(agents, transcript, seed: str, model: str, turns: int, path: Path):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# TinyTroupe Conversation — {ts}",
        "",
        f"- **Model:** `{model}`",
        f"- **Turns:** {turns}",
        f"- **Seed:** {seed}",
        "",
        "## Participants",
    ]
    for a in agents:
        lines.append(f"- **{a.persona.name}** — {a.persona.role}")
    lines += ["", "## Transcript", ""]

    for row in transcript:
        lines.append(f"**Turn {row['turn']} — {row['speaker']} ({row['role']})**")
        lines.append(row["text"])
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")

if __name__ == "__main__":
    agents, transcript = run_simulation(SEED, MAX_TURNS)
    write_jsonl(transcript, LOG_PATH)
    write_markdown(agents, transcript, SEED, MODEL, MAX_TURNS, README_PATH)
    print(f"Wrote:\n  - {LOG_PATH}\n  - {README_PATH}")
