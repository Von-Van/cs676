import os
import json
import time
import random
import copy
import configparser
from pathlib import Path
from datetime import datetime, timedelta

# TinyTroupe
from tinytroupe.agent import TinyPerson
from tinytroupe.environment import TinyWorld
from tinytroupe import config_manager
from tinytroupe.openai_utils import force_api_cache

# --------------------
# Paths & config
# --------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
CFG_PATH = ROOT / "config" / "config.ini"

OUT_DIR.mkdir(parents=True, exist_ok=True)

def normalize_log_level(level: str) -> str:
    """
    Normalize and validate log level string.
    Returns uppercase version of valid log level or 'INFO' as fallback.
    """
    VALID_LEVELS = {'DEBUG', 'INFO', 'WARNING', 'WARN', 'ERROR', 'CRITICAL', 'FATAL'}
    
    if not level:
        return 'INFO'
    
    # Convert to uppercase and handle common variants
    normalized = level.upper().strip()
    if normalized == 'WARN':
        normalized = 'WARNING'
    elif normalized == 'FATAL':
        normalized = 'CRITICAL'
    
    # Validate against known good levels
    if normalized not in VALID_LEVELS:
        print(f"Warning: Unknown log level '{level}', using 'INFO'")
        return 'INFO'
    
    return normalized

CFG = configparser.ConfigParser()
if CFG_PATH.exists():
    CFG.read(CFG_PATH)
else:
    raise FileNotFoundError(f"Missing config at {CFG_PATH}")

# Mirror canonical config sections to TinyTroupe at runtime
if CFG.has_section("OpenAI"):
    if CFG.has_option("OpenAI", "MODEL"):
        config_manager.update("model", CFG.get("OpenAI", "MODEL"))

# API Key Security: Priority order for retrieving OpenAI API key
# 1. Streamlit secrets (recommended for deployed apps)
# 2. Environment variable (recommended for local dev)
# 3. Config file (fallback, less secure - avoid committing keys)
API_KEY = None
try:
    import streamlit as st
    if hasattr(st, 'secrets') and 'openai' in st.secrets and 'api_key' in st.secrets['openai']:
        API_KEY = st.secrets['openai']['api_key']
except (ImportError, FileNotFoundError, KeyError):
    pass

if not API_KEY:
    API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    API_KEY = CFG.get("openai", "api_key", fallback="")

if not API_KEY:
    raise RuntimeError(
        "OpenAI API key not found. Please set via one of:\n"
        "  1. Streamlit secrets: .streamlit/secrets.toml (see secrets.toml.example)\n"
        "  2. Environment variable: OPENAI_API_KEY\n"
        "  3. Config file: config/config.ini [openai] api_key (less secure)"
    )

MODEL = CFG.get("model", "name", fallback="gpt-4.1-mini")
TEMPERATURE = CFG.getfloat("run", "temperature", fallback=0.7)
MAX_TURNS = CFG.getint("run", "max_turns", fallback=6)
SEED = CFG.get("run", "seed", fallback=(
    "Debate whether to ship an auto-context feature that summarizes the last 3 minutes "
    "of a livestream for late-joining viewers."
))

PERSONAS_PATH = DATA_DIR / "personas.agents.json"
if not PERSONAS_PATH.exists():
    raise FileNotFoundError(f"Missing personas file at {PERSONAS_PATH}")

LOG_PATH = OUT_DIR / "conversation_log.jsonl"
README_PATH = OUT_DIR / "conversation_readme.md"

force_api_cache(True)  # or set in config.ini as CACHE_API_CALLS=True

# --------------------
# Helpers
# --------------------

def backoff_sleep(attempt: int, base: float = 0.8) -> None:
    # jittered exponential backoff
    delay = base * (2 ** attempt) + random.uniform(0, 0.25)
    time.sleep(delay)

def load_personas(path: Path):
    """
    Load personas from JSON and create TinyPerson agents with proper initialization.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    agents = []
    for spec in data:
        if not isinstance(spec, dict) or "type" not in spec or spec["type"] != "TinyPerson":
            raise ValueError(f"Invalid persona specification: {spec}")
        
        # Ensure required fields
        required_fields = ["id", "role", "description", "personality", "prompting"]
        missing = [f for f in required_fields if f not in spec]
        if missing:
            raise ValueError(f"Missing required fields in persona: {missing}")
        
        # Copy spec and ensure model settings
        agent_spec = spec.copy()
        
        # Create agent with id as name since TinyPerson uses name as primary identifier
        try:
            # Initialize TinyPerson directly with required fields
            agent = TinyPerson(name=agent_spec["id"])
            
            # Configure the persona details
            role_value = agent_spec["role"]
            agent.define("role", role_value)
            agent.define("description", agent_spec["description"])

            # Populate basic biography fields expected by TinyPerson internals
            # occupation is used by minibio; set it from role when missing
            agent.define("occupation", agent_spec.get("occupation", role_value))
            agent.define("nationality", agent_spec.get("nationality", None))
            agent.define("country_of_residence", agent_spec.get("country_of_residence", None))
            # ensure there is a 'residence' key to avoid KeyError in older TinyPerson code
            residence_fallback = agent_spec.get("residence") or agent_spec.get("country_of_residence") or agent_spec.get("nationality") or "Unknown"
            agent.define("residence", residence_fallback)
            agent.define("age", agent_spec.get("age", None))

            # Configure personality as a nested dict (TinyPerson expects persona.personality)
            personality = agent_spec.get("personality", {})
            if isinstance(personality, dict):
                agent.define("personality", personality)
            
            # Set model parameters
            agent.define("model", agent_spec.get("model", MODEL))
            agent.define("temperature", agent_spec.get("temperature", TEMPERATURE))
            
            # Configure prompting
            if "prompting" in agent_spec:
                agent.define("prompting", agent_spec["prompting"])
            
            # Store any additional fields
            for key, value in agent_spec.items():
                if key not in ["type", "id", "role", "description", "personality", "model", "temperature", "prompting"]:
                    agent.define(key, value)
            
            agents.append(agent)
            
        except Exception as e:
            raise ValueError(f"Failed to create persona {spec.get('id', 'unknown')}: {str(e)}")
    
    if not agents:
        raise ValueError("No valid personas found in configuration")
    
    return agents

def _iter_agents(world):
    """
    Robustly iterate world occupants across TinyTroupe versions.
    Prefers 'agents' (documented), but falls back gracefully.
    """
    for attr in ("agents", "residents", "actors", "population", "members"):
        val = getattr(world, attr, None)
        if isinstance(val, (list, tuple)):
            return list(val)
    raise AttributeError("Could not find agent list on TinyWorld")

# --------------------
# Run simulation
# --------------------
def run_simulation(seed: str, turns: int):
    """
    Run a discussion simulation with the given seed topic for the specified number of turns.
    Each turn allows every agent to contribute once to the discussion.
    """
    # Load and initialize agents
    agents = load_personas(PERSONAS_PATH)

    # Initialize world with current datetime and agents
    world = TinyWorld(
        name="DiscussionRoom",
        agents=agents,
        initial_datetime=datetime.now()
    )

    # Make everyone visible to each other in the discussion
    world.make_everyone_accessible()
    
    # Set initial context and location for all agents
    world.broadcast_context_change([
        "Group discussion about product features and technology",
        f"Topic: {seed}"
    ])
    
    for agent in agents:
        agent.move_to("DiscussionRoom")
        
    # Set initial discussion goal
    world.broadcast_internal_goal(
        "Participate in a constructive discussion about the proposed feature, "
        "considering technical feasibility, user value, and potential challenges."
    )
    
    # Start the discussion with the seed topic
    world.broadcast(f"Let's begin our discussion on an important topic: {seed}", source=world)

    transcript = []

    def safe_to_text(obj) -> str:
        """Convert common content objects to plain strings to avoid library-specific objects (e.g., Document)."""
        if obj is None:
            return ""
        if isinstance(obj, str):
            return obj
        # Common 'Document' or similar objects expose .text as read-only attribute
        try:
            txt = getattr(obj, 'text', None)
            if isinstance(txt, str) and txt:
                return txt
        except Exception:
            pass
        # If dict-like, try common keys
        if isinstance(obj, dict):
            for key in ("content", "text", "message", "body"):
                val = obj.get(key)
                if isinstance(val, str):
                    return val
        # Fallback to JSON if possible, else str()
        try:
            return json.dumps(obj, ensure_ascii=False, default=str)
        except Exception:
            try:
                return str(obj)
            except Exception:
                return ""
    
    # Run the simulation for the specified number of turns
    # Each turn represents ~15 minutes in simulation time
    actions_over_time = world.run(
        steps=turns,
        timedelta_per_step=timedelta(minutes=15),
        parallelize=True,  # Allow parallel agent actions
        randomize_agents_order=True,  # Randomize speaking order each turn
        return_actions=True  # Get all actions for transcript
    )
    
    # Process actions into transcript (robust to a few return formats)
    for turn, turn_actions in enumerate(actions_over_time, 1):
        for agent_name, actions in (turn_actions or {}).items():
            agent = world.get_agent_by_name(agent_name)

            if not actions:
                continue

            # actions may be a list of content dicts, or a single dict
            if isinstance(actions, dict):
                actions_iter = [actions]
            else:
                actions_iter = list(actions)

            for action_item in actions_iter:
                # action_item is often a content dict that contains an 'action' key
                if not isinstance(action_item, dict):
                    continue

                # Try common shapes: {'action': {...}}, or {'type':..., 'content':...}
                if "action" in action_item and isinstance(action_item["action"], dict):
                    act = action_item["action"]
                else:
                    act = action_item if ("type" in action_item and "content" in action_item) else None

                if not act:
                    # Sometimes content is nested under other keys; try common alternatives
                    act = action_item.get("message") or action_item.get("payload")

                if not isinstance(act, dict):
                    continue

                a_type = act.get("type", "").upper()
                if a_type in ("TALK", "CONVERSATION") or (a_type == "" and isinstance(act.get("content", ""), str)):
                    raw_content = act.get("content") or ""
                    content = safe_to_text(raw_content).strip()
                    if content:
                        # Broadcast content to other agents as plain text to avoid passing Document-like objects
                        for other in agents:
                            if other != agent:
                                try:
                                    other.listen(content, source=agent)
                                except Exception:
                                    # ignore individual listener errors here; environment will handle
                                    pass

                        transcript.append({
                            "turn": turn,
                            "speaker": agent_name,
                            "role": agent._persona.get("role", "Unknown") if hasattr(agent, "_persona") else getattr(agent, "role", "Unknown"),
                            "text": content
                        })

    return agents, transcript

def write_jsonl(rows, path: Path):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_markdown(agents, transcript, seed: str, model: str, turns: int, path: Path):
    """Generate a detailed Markdown report of the simulation run."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    # Collect runtime settings
    is_parallel = config_manager.get("parallel", False)
    cache_enabled = config_manager.get("cache_api_calls", False)
    
    lines = [
        f"# TinyTroupe Conversation — {ts}",
        "",
        "## Configuration",
        "",
        "### Runtime Settings",
        f"- **Model:** `{model}`",
        f"- **Temperature:** `{TEMPERATURE}`",
        f"- **Parallel Execution:** `{is_parallel}`",
        f"- **API Cache:** `{cache_enabled}`",
        "",
        "### Simulation Parameters",
        f"- **Turns:** {turns}",
        f"- **Agents:** {len(agents)}",
        f"- **Messages:** {len(transcript)}",
        "",
        "### Initial Prompt",
        "```",
        seed,
        "```",
        "",
        "## Participants",
        "",
    ]

    # Add detailed agent information
    for a in agents:
        role = a._persona.get("role", "Unknown") if hasattr(a, "_persona") else getattr(a, "role", "Unknown")
        lines.extend([
            f"### {a.name}",
            f"**Role:** {role}",
        ])

        # Personality is stored inside the persona dict
        personality = a._persona.get("personality", {}) if hasattr(a, "_persona") else {}

        traits = personality.get("traits", []) if isinstance(personality, dict) else []
        if traits:
            lines.append("**Traits:**")
            for trait in traits:
                lines.append(f"- {trait}")

        goals = personality.get("goals", []) if isinstance(personality, dict) else []
        if goals:
            lines.append("**Goals:**")
            for goal in goals:
                lines.append(f"- {goal}")

        style = personality.get("style") if isinstance(personality, dict) else None
        if style:
            lines.append(f"**Style:** {style}")
        
        lines.append("")  # spacing between agents
    
    # Add transcript with turn grouping
    lines.extend(["## Transcript", ""])
    current_turn = None
    
    for row in transcript:
        turn = row["turn"]
        if turn != current_turn:
            current_turn = turn
            lines.append(f"### Turn {turn}")
            lines.append("")
        
        lines.extend([
            f"**{row['speaker']} ({row['role']}):**",
            row["text"],
            ""
        ])
    
    # Add metadata footer
    lines.extend([
        "---",
        "## About this Transcript",
        f"- **Generated:** {ts}",
        f"- **Script:** `{Path(__file__).name}`",
        f"- **Persona File:** `{PERSONAS_PATH.name}`",
        "",
        "For experiment runs with multiple trials or variants, see `run_experiment.py`."
    ])
    
    path.write_text("\n".join(lines), encoding="utf-8")

if __name__ == "__main__":
    agents, transcript = run_simulation(SEED, MAX_TURNS)
    write_jsonl(transcript, LOG_PATH)
    write_markdown(agents, transcript, SEED, MODEL, MAX_TURNS, README_PATH)
    print(f"Wrote:\n  - {LOG_PATH}\n  - {README_PATH}")
