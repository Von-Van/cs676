# app.py
import json
import time
from pathlib import Path

import streamlit as st

# Import your simulation helpers & constants
import scripts.run_simulation as sim
from tinytroupe.environment import TinyWorld
from tinytroupe.openai_utils import force_api_cache

# ---------- App Setup ----------
st.set_page_config(page_title="TinyTroupe Persona Simulator — Beta", layout="wide")

st.title("🧠 TinyTroupe Persona Simulator — Beta")
st.markdown(
    """
    Welcome to the TinyTroupe Persona Simulator! This interactive tool lets you configure and run persona-based discussions for product ideation, feature evaluation, and scenario exploration.
    
    **How to use:**
    1. Configure simulation parameters in the sidebar (model, temperature, turns, personas, etc.).
    2. Enter a discussion topic or feature prompt below.
    3. Click **Run Simulation** to generate a transcript.
    4. View, explore, and download the results.
    """
)

DATA_DIR = Path(sim.DATA_DIR) if hasattr(sim, "DATA_DIR") else Path("data")
PERSONAS_JSON = DATA_DIR / "personas.agents.json"

# ---------- Sidebar Controls ----------
st.sidebar.header("Simulation Settings")


# Model & temperature
model = st.sidebar.selectbox(
    "Model",
    options=["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o", sim.MODEL],
    index=0,
    help="Choose the language model for simulation. Smaller models are faster and less costly."
)
temperature = st.sidebar.slider(
    "Temperature",
    0.0, 1.5, float(sim.TEMPERATURE), 0.1,
    help="Higher values = more creative, lower = more predictable."
)

# Turns
turns = st.sidebar.slider(
    "Number of Turns",
    1, 12, int(sim.MAX_TURNS),
    help="How many conversation rounds the personas will go through."
)

# Caching
use_cache = st.sidebar.checkbox(
    "Cache API calls (faster reruns, cheaper)",
    value=True,
    help="Enable to reduce API cost and speed up repeated runs."
)

# Load persona specs
raw_specs = json.loads(PERSONAS_JSON.read_text(encoding="utf-8"))
all_names = [p.get("name", f"Agent {i+1}") for i, p in enumerate(raw_specs)]
selected_names = st.sidebar.multiselect(
    "Personas to Include",
    options=all_names,
    default=all_names,
    help="Select which personas will participate in the simulation."
)
# Optional: pick N out of selected
limit_n = st.sidebar.number_input(
    "Limit # of Personas (0 = all)",
    0, len(selected_names), 0,
    help="Limit the number of personas included. 0 = use all selected."
)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: Fewer personas + smaller model = fewer rate limits.")

# ---------- Main Controls ----------

seed = st.text_area(
    "Discussion Topic / Feature Prompt",
    value=sim.SEED,
    height=120,
    help="Describe the feature, scenario, or product idea for the personas to discuss."
)

col_run, col_clear = st.columns([1, 1])
run_clicked = col_run.button("▶️ Run Simulation", type="primary")
clear_clicked = col_clear.button("🧹 Clear Output")

if clear_clicked:
    st.session_state.pop("transcript", None)
    st.session_state.pop("agents", None)
    st.rerun()

# ---------- Helper: run a subset simulation (UI uses this) ----------
def simulate_subset(seed_text: str, n_turns: int, persona_specs: list[dict]):
    """Light wrapper that mirrors your run_simulation() logic but on a subset."""
    # Apply UI overrides to your module globals so helpers use them
    sim.MODEL = model
    sim.TEMPERATURE = float(temperature)

    # Build agents from specs
    agents = [sim.make_agent_from_template(**p) for p in persona_specs]

    world = TinyWorld(name="UIRoom", agents=agents)
    try:
        world.make_everyone_accessible()
    except Exception:
        pass

    world.broadcast(seed_text)
    transcript = []

    # Simple progress indicator
    total_steps = n_turns * len(agents)
    progress = st.progress(0)
    step = 0

    for turn in range(1, n_turns + 1):
        for agent in getattr(world, "agents", agents):
            last_err = None
            for attempt in range(6):
                try:
                    msg = agent.listen_and_act(
                        "Contribute one concise, concrete point and hand off implicitly."
                    )
                    break
                except Exception as e:
                    s = str(e).lower()
                    if "429" in s or "rate" in s or "temporar" in s:
                        wait = min(5 * (2 ** attempt), 60)
                        st.caption(f"Rate limit hit — waiting {wait}s …")
                        time.sleep(wait)
                        last_err = e
                        continue
                    raise
            else:
                raise RuntimeError(f"Backoff exhausted on model call: {last_err}") from last_err

            text = msg if isinstance(msg, str) else getattr(msg, "text", str(msg))
            transcript.append({
                "turn": turn,
                "speaker": sim._get_persona_field(agent, "name"),
                "role": sim._get_persona_field(agent, "role"),
                "text": (text or "").strip()
            })

            step += 1
            progress.progress(min(step / max(total_steps, 1), 1.0))

            # Gentle pacing to avoid bursts
            time.sleep(1)

    return agents, transcript

# ---------- Run ----------
if run_clicked:
    # Cache toggle
    force_api_cache(bool(use_cache))

    # Filter persona list
    chosen = [p for p in raw_specs if p.get("name") in selected_names]
    if limit_n and limit_n < len(chosen):
        chosen = chosen[:limit_n]

    if not chosen:
        st.error("⚠️ Please select at least one persona to run the simulation.")
    elif not seed.strip():
        st.error("⚠️ Please enter a discussion topic or feature prompt.")
    else:
        with st.spinner("Running simulation…"):
            agents, transcript = simulate_subset(seed, turns, chosen)
            st.session_state["agents"] = agents
            st.session_state["transcript"] = transcript

# ---------- Show Output ----------
agents = st.session_state.get("agents")
transcript = st.session_state.get("transcript")

if transcript:
    st.success(f"Simulation complete — {len(transcript)} messages.")
    st.subheader("Transcript")
    for row in transcript:
        with st.chat_message(row["speaker"]):
            st.markdown(f"**({row['role']})** {row['text']}")

    # Downloads
    jsonl_data = "\n".join(json.dumps(r, ensure_ascii=False) for r in transcript)
    md_lines = [
        f"# Conversation",
        f"- **Model:** `{model}`",
        f"- **Turns:** {turns}",
        f"- **Seed:** {seed}",
        "",
        "## Transcript",
        ""
    ]
    for r in transcript:
        md_lines.append(f"**Turn {r['turn']} — {r['speaker']} ({r['role']})**")
        md_lines.append(r["text"])
        md_lines.append("")
    md_data = "\n".join(md_lines)

    c1, c2 = st.columns(2)
    c1.download_button("💾 Download .jsonl", data=jsonl_data, file_name="conversation_log.jsonl", mime="application/json")
    c2.download_button("📝 Download .md", data=md_data, file_name="conversation_readme.md", mime="text/markdown")

# ---------- Footer ----------
st.markdown("---")
st.caption(f"Model: `{model}` • Temperature: {temperature} • Turns: {turns} • Personas: {len(selected_names) if selected_names else 0}")
