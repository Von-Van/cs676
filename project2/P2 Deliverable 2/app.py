# app.py
# Streamlit entry point for the TinyTroupe-based product simulation tool.
# Architecture at a glance:
# - UI layer (this file) collects parameters, personas, and environment settings.
# - Backend helpers (scripts/run_simulation.py) provide model defaults and
#   persona construction logic. We reuse its schema expectations.
# - simulate_subset() below adapts those helpers to the interactive UI by
#   assembling a subset of personas (including user-defined ones) and creating
#   a TinyWorld with optional environment configuration. We then step through
#   turns, capturing messages for the transcript.
import json
import time
from pathlib import Path
from datetime import datetime

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

# ---------- Rate Limiting & Cost Controls ----------
# Initialize session state for security controls
if "run_count" not in st.session_state:
    st.session_state["run_count"] = 0
if "last_run_time" not in st.session_state:
    st.session_state["last_run_time"] = None
if "total_api_calls" not in st.session_state:
    st.session_state["total_api_calls"] = 0
# Interactive mode state
if "active_world" not in st.session_state:
    st.session_state["active_world"] = None
if "active_agents" not in st.session_state:
    st.session_state["active_agents"] = None
if "interaction_history" not in st.session_state:
    st.session_state["interaction_history"] = []
if "interaction_mode" not in st.session_state:
    st.session_state["interaction_mode"] = "group"  # "group" or "individual"

# Rate limiting configuration
MAX_RUNS_PER_HOUR = 10  # Adjust based on your needs
COOLDOWN_SECONDS = 30  # Minimum time between runs
MAX_API_CALLS_PER_SESSION = 1000  # Protect against runaway costs

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

# Add custom agent names to the list of available personas
custom_agent_names = [a.get("name") for a in st.session_state.get("custom_agents", [])]
all_names_with_custom = all_names + custom_agent_names

selected_names = st.sidebar.multiselect(
    "Personas to Include",
    options=all_names_with_custom,
    default=all_names_with_custom,  # Select all by default, including custom ones
    help="Select which personas will participate in the simulation."
)
# Optional: pick N out of selected
limit_n = st.sidebar.number_input(
    "Limit # of Personas (0 = all)",
    0, len(selected_names), 0,
    help="Limit the number of personas included. 0 = use all selected."
)

st.sidebar.markdown("---")

# Security & Cost Controls Panel (after we have turns and selected_names)
with st.sidebar.expander("🔒 Security & Limits", expanded=False):
    st.caption("Rate limiting and cost controls")
    st.metric("Runs this session", st.session_state["run_count"])
    st.metric("API calls this session", st.session_state["total_api_calls"])
    
    # Estimate tokens for current configuration
    num_selected = len(selected_names)
    est_tokens_per_turn = num_selected * 500  # Rough estimate: 500 tokens per agent per turn
    est_total_tokens = est_tokens_per_turn * turns
    est_cost = (est_total_tokens / 1000) * 0.002  # Rough cost estimate for gpt-4o-mini
    
    st.metric("Est. tokens (next run)", f"{est_total_tokens:,}")
    st.metric("Est. cost (next run)", f"${est_cost:.4f}")
    
    if st.session_state["total_api_calls"] > MAX_API_CALLS_PER_SESSION * 0.8:
        st.warning(f"⚠️ Approaching session limit ({MAX_API_CALLS_PER_SESSION} API calls)")

st.sidebar.markdown("---")
st.sidebar.caption("Tip: Fewer personas + smaller model = fewer rate limits.")

# ---------- Customization Panels ----------
# We provide two expanders:
# 1) Agent customization: create ad-hoc agents without editing the JSON file.
# 2) Environment customization: tweak world name, context, goal, pacing.

# Init session state containers the first time
if "custom_agents" not in st.session_state:
    st.session_state["custom_agents"] = []  # list of persona-spec dicts following TinyTroupe schema

with st.sidebar.expander("➕ Create or Customize Agent", expanded=False):
    st.caption("Define a new simulated individual. Saved only for this session.")
    ca_name = st.text_input("Display Name", help="How this agent appears in the transcript.")
    ca_id = st.text_input("Unique ID", help="Internal identifier (letters, numbers, underscores). If empty, we'll derive from name.")
    ca_role = st.text_input("Role", value="Product Manager")
    ca_desc = st.text_area("Short Bio / Description", height=80)
    ca_traits = st.text_input("Personality Traits (comma-separated)", help="e.g., analytical,pragmatic,user-obsessed")
    ca_goals = st.text_area("Goals (one per line)", height=70)
    ca_style = st.text_input("Speaking Style", value="Concise and actionable")
    ca_model = st.selectbox("Agent Model (override)", options=["", "gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o", sim.MODEL], index=0, help="Leave blank to use global model above.")
    ca_temp = st.slider("Agent Temperature (override)", 0.0, 1.5, value=float(sim.TEMPERATURE), step=0.1)
    ca_prompting = st.text_area("Prompting notes (optional)", value="Implicit handoff. Keep it short.", height=60)
    add_agent = st.button("Add Custom Agent")

    if add_agent:
        # Assemble spec matching scripts.run_simulation.load_personas() expectations
        uid = (ca_id or ca_name or "custom_agent").strip().replace(" ", "_")
        traits_list = [t.strip() for t in (ca_traits or "").split(",") if t.strip()]
        goals_list = [g.strip() for g in (ca_goals or "").splitlines() if g.strip()]
        spec = {
            "type": "TinyPerson",
            "id": uid,
            "name": ca_name or uid,
            "role": ca_role or "Contributor",
            "description": ca_desc or "",
            "personality": {
                "traits": traits_list,
                "goals": goals_list,
                "style": ca_style or ""
            },
            "model": ca_model or model,
            "temperature": ca_temp,
            "prompting": {"notes": ca_prompting or ""}
        }
        st.session_state["custom_agents"].append(spec)
        st.success(f"Added custom agent: {spec['name']}")
        # Trigger rerun to update the "Personas to Include" multiselect
        st.rerun()

include_custom_agents = False
custom_names = [a.get("name") for a in st.session_state["custom_agents"]]
if custom_names:
    st.sidebar.caption(f"✓ {len(custom_names)} custom agent(s) available")
    # Note: Custom agents are now included in the main "Personas to Include" multiselect above

with st.sidebar.expander("🌍 Environment Settings", expanded=False):
    st.caption("Tune the world in which agents interact.")
    env_world_name = st.text_input("World Name", value="UIRoom")
    env_room_name = st.text_input("Room/Location", value="DiscussionRoom")
    env_initial_dt_mode = st.selectbox("Initial Time", options=["Now", "Custom ISO"], index=0)
    env_initial_dt_str = st.text_input("Custom ISO (YYYY-MM-DD HH:MM)", value="", help="Used only if 'Custom ISO' is selected")
    env_context = st.text_area("Context (one per line)", value="Group discussion about product features and technology", height=60)
    env_goal = st.text_area("Shared Internal Goal", value="Participate constructively. Be concrete and practical.", height=60)
    pacing_seconds = st.slider("Pacing between turns (seconds)", 0.0, 3.0, value=1.0, step=0.5)


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
    st.session_state.pop("active_world", None)
    st.session_state.pop("active_agents", None)
    st.session_state.pop("interaction_history", None)
    st.rerun()

# ---------- Helper: run a subset simulation (UI uses this) ----------
def simulate_subset(seed_text: str, n_turns: int, persona_specs: list[dict]):
    """Run a simulation on a subset of personas.

    Data Flow Summary:
    1) persona_specs: filtered base personas from file.
    2) session custom agents optionally appended (same schema).
    3) sanitized -> temp JSON -> load_personas() for canonical TinyPerson creation.
    4) environment settings applied to TinyWorld (name, datetime, context, goal).
    5) turn loop: each agent acts once per turn; we suppress memory consolidation
       to mitigate library issue; transcript captures raw output.
    """
    # Apply UI overrides to your module globals so helpers use them
    sim.MODEL = model
    sim.TEMPERATURE = float(temperature)

    # Always create fresh agent objects for each run to avoid name collision
    all_specs = json.loads(PERSONAS_JSON.read_text(encoding="utf-8"))
    
    # Merge base personas with custom personas
    # Include custom agents that were selected in the multiselect
    custom_agents = st.session_state.get("custom_agents", [])
    all_specs_combined = all_specs + custom_agents
    
    selected_specs = [spec for spec in all_specs_combined if spec.get("name") in [p.get("name") for p in persona_specs]]
    if limit_n and limit_n < len(selected_specs):
        selected_specs = selected_specs[:limit_n]
    # Deep recursive sanitization: ensure all fields are plain strings
    def deep_sanitize(obj):
        if isinstance(obj, dict):
            return {k: deep_sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [deep_sanitize(item) for item in obj]
        elif obj is None:
            return ""
        else:
            return str(obj)
    selected_specs = [deep_sanitize(spec) for spec in selected_specs]
    # Use sim.load_personas logic for selected specs
    import copy
    temp_path = DATA_DIR / "_temp_selected_personas.json"
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(selected_specs, f, ensure_ascii=False)
    agents = sim.load_personas(temp_path)
    temp_path.unlink(missing_ok=True)

    # Sanitize prompt
    safe_seed_text = deep_sanitize(seed_text)
    # Resolve initial datetime
    if 'env_initial_dt_mode' in globals():
        pass  # placeholder to appease static tools
    init_dt = datetime.now()
    if env_initial_dt_mode == "Custom ISO" and env_initial_dt_str.strip():
        try:
            init_dt = datetime.fromisoformat(env_initial_dt_str.strip())
        except Exception:
            st.warning("Invalid custom datetime; falling back to now().")

    world = TinyWorld(name=env_world_name or "UIRoom", agents=agents, initial_datetime=init_dt)
    try:
        world.make_everyone_accessible()
    except Exception:
        pass

    # Apply environment context & goal
    context_lines = [l.strip() for l in (env_context or "").splitlines() if l.strip()]
    if context_lines:
        try:
            world.broadcast_context_change(context_lines)
        except Exception:
            # Fallback: broadcast each line
            for ln in context_lines:
                try:
                    world.broadcast(ln)
                except Exception:
                    pass
    if env_goal.strip():
        try:
            world.broadcast_internal_goal(env_goal.strip())
        except Exception:
            pass

    # Move agents to chosen room (if provided)
    for a in agents:
        try:
            a.move_to(env_room_name or "DiscussionRoom")
        except Exception:
            pass

    world.broadcast(safe_seed_text)
    transcript = []

    total_steps = n_turns * len(agents)
    progress = st.progress(0)
    step = 0

    def safe_to_text(obj) -> str:
        """Convert TinyTroupe return types (dict/list/Document/etc.) to plain text."""
        if obj is None:
            return ""
        if isinstance(obj, str):
            return obj
        # If it's a list, join text-like items
        if isinstance(obj, list):
            parts = [safe_to_text(x) for x in obj]
            return "\n\n".join([p for p in parts if p])
        # Dict with common content shapes
        if isinstance(obj, dict):
            # common keys
            for key in ("content", "text", "message", "body"):
                if key in obj and isinstance(obj[key], (str, dict, list)):
                    return safe_to_text(obj[key])
            # nested 'action' field
            if "action" in obj and isinstance(obj["action"], dict):
                return safe_to_text(obj["action"].get("content"))
        # Objects with .text attribute
        try:
            txt = getattr(obj, "text", None)
            if isinstance(txt, str):
                return txt
        except Exception:
            pass
        # Fallback
        try:
            return json.dumps(obj, ensure_ascii=False, default=str)
        except Exception:
            return str(obj)

    def get_role(agent_obj) -> str:
        # TinyPerson typically keeps role in _persona
        try:
            if hasattr(agent_obj, "_persona") and isinstance(agent_obj._persona, dict):
                return agent_obj._persona.get("role") or getattr(agent_obj, "role", "Unknown")
        except Exception:
            pass
        return getattr(agent_obj, "role", "Unknown") or "Unknown"

    for turn in range(1, n_turns + 1):
        for agent in getattr(world, "agents", agents):
            last_err = None
            for attempt in range(6):
                # Temporarily disable memory consolidation to avoid Document.text setter errors
                original_consolidate = getattr(agent, "consolidate_episode_memories", None)
                if callable(original_consolidate):
                    try:
                        # simple no-op replacement
                        agent.consolidate_episode_memories = lambda: None
                    except Exception:
                        original_consolidate = None
                try:
                    msg = agent.listen_and_act(
                        "Contribute one concise, concrete point and hand off implicitly.",
                        return_actions=True
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
                finally:
                    if callable(original_consolidate):
                        try:
                            agent.consolidate_episode_memories = original_consolidate
                        except Exception:
                            pass
            else:
                raise RuntimeError(f"Backoff exhausted on model call: {last_err}") from last_err

            text = safe_to_text(msg).strip()
            transcript.append({
                "turn": turn,
                "speaker": getattr(agent, "name", "Unknown"),
                "role": get_role(agent),
                "text": (text or "").strip()
            })

            step += 1
            progress.progress(min(step / max(total_steps, 1), 1.0))
            time.sleep(pacing_seconds)

    # Set session state for output
    st.session_state["agents"] = agents
    st.session_state["transcript"] = transcript
    # Store active world and agents for interactive mode
    st.session_state["active_world"] = world
    st.session_state["active_agents"] = agents
    return agents, transcript


# ---------- Run ----------
if run_clicked:
    # Rate limiting check
    current_time = time.time()
    
    # Check cooldown period
    if st.session_state["last_run_time"]:
        time_since_last_run = current_time - st.session_state["last_run_time"]
        if time_since_last_run < COOLDOWN_SECONDS:
            remaining = int(COOLDOWN_SECONDS - time_since_last_run)
            st.error(f"⚠️ Please wait {remaining} seconds before starting another simulation.")
            st.stop()
    
    # Check hourly rate limit (approximate with session limit)
    if st.session_state["run_count"] >= MAX_RUNS_PER_HOUR:
        st.error(f"⚠️ Session limit reached ({MAX_RUNS_PER_HOUR} runs). Please refresh the page to reset.")
        st.stop()
    
    # Check API call limit
    if st.session_state["total_api_calls"] >= MAX_API_CALLS_PER_SESSION:
        st.error(f"⚠️ API call limit reached ({MAX_API_CALLS_PER_SESSION} calls). Please refresh to reset.")
        st.stop()
    
    # Cost control: validate agent and turn limits
    num_agents = len(selected_names)
    # Custom agents are already included in selected_names, no need to add separately
    
    MAX_AGENTS = 10  # Configurable limit
    MAX_TURNS_LIMIT = 12  # Already enforced by slider, but double-check
    
    if num_agents > MAX_AGENTS:
        st.error(f"⚠️ Too many agents selected ({num_agents}). Maximum allowed: {MAX_AGENTS}")
        st.stop()
    
    if turns > MAX_TURNS_LIMIT:
        st.error(f"⚠️ Too many turns ({turns}). Maximum allowed: {MAX_TURNS_LIMIT}")
        st.stop()
    
    # Cache toggle
    force_api_cache(bool(use_cache))
    # Decision: caching is applied globally before we construct agents so that
    # underlying TinyTroupe OpenAI calls benefit across all agent actions.

    # Filter persona list
    # Architecture note: We take the base personas from JSON and optionally append
    # in-memory session custom agents (already schema-compatible). This ensures
    # the canonical loader path (load_personas) stays the single source of truth
    # for constructing TinyPerson instances.
    chosen = [p for p in raw_specs if p.get("name") in selected_names]
    if limit_n and limit_n < len(chosen):
        chosen = chosen[:limit_n]

    if not chosen:
        st.error("⚠️ Please select at least one persona to run the simulation.")
    elif not seed.strip():
        st.error("⚠️ Please enter a discussion topic or feature prompt.")
    else:
        with st.spinner("Running simulation…"):
            # Track simulation start
            st.session_state["last_run_time"] = time.time()
            st.session_state["run_count"] += 1
            
            # Estimate API calls (agents * turns)
            estimated_calls = num_agents * turns
            st.session_state["total_api_calls"] += estimated_calls
            
            agents, transcript = simulate_subset(seed, turns, chosen)
            st.session_state["agents"] = agents
            st.session_state["transcript"] = transcript

# ---------- Show Output ----------
agents = st.session_state.get("agents")
transcript = st.session_state.get("transcript")

if transcript:
    st.success(f"Simulation complete — {len(transcript)} messages.")
    st.subheader("Transcript")
    # Rendering choice: Streamlit chat_message provides a more conversational UI.
    # We include role for quick situational awareness, but keep utterances concise.
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

    # ---------- Interactive Mode ----------
    # Real-time interaction: allow users to ask follow-up questions and explore feature feedback dynamically
    st.markdown("---")
    st.subheader("💬 Interactive Follow-Up")
    st.caption("Ask follow-up questions, probe deeper, or explore different aspects of the discussion.")
    
    # Check if we have an active world to interact with
    if st.session_state.get("active_world") and st.session_state.get("active_agents"):
        
        # Mode selector: group or individual
        col_mode, col_target = st.columns([1, 2])
        with col_mode:
            interaction_mode = st.radio(
                "Interaction Mode",
                options=["Group Discussion", "Ask Individual"],
                index=0 if st.session_state["interaction_mode"] == "group" else 1,
                help="Choose whether to ask the group or target a specific persona"
            )
            st.session_state["interaction_mode"] = "group" if interaction_mode == "Group Discussion" else "individual"
        
        # Target persona selector (only for individual mode)
        target_agent = None
        with col_target:
            if st.session_state["interaction_mode"] == "individual":
                agent_names = [a.name for a in st.session_state["active_agents"]]
                target_name = st.selectbox(
                    "Select Persona",
                    options=agent_names,
                    help="Choose which persona to ask"
                )
                target_agent = next((a for a in st.session_state["active_agents"] if a.name == target_name), None)
        
        # Chat input
        user_question = st.chat_input("Ask a follow-up question...")
        
        if user_question:
            # Rate limit check for interactions
            if st.session_state["total_api_calls"] >= MAX_API_CALLS_PER_SESSION:
                st.error(f"⚠️ API call limit reached ({MAX_API_CALLS_PER_SESSION} calls). Please refresh to reset.")
            else:
                with st.spinner("Getting response..."):
                    # Handle the interaction based on mode
                    world = st.session_state["active_world"]
                    
                    def safe_to_text(obj) -> str:
                        """Convert TinyTroupe return types to plain text."""
                        if obj is None:
                            return ""
                        if isinstance(obj, str):
                            return obj
                        if isinstance(obj, list):
                            parts = [safe_to_text(x) for x in obj]
                            return "\n\n".join([p for p in parts if p])
                        if isinstance(obj, dict):
                            for key in ("content", "text", "message", "body"):
                                if key in obj and isinstance(obj[key], (str, dict, list)):
                                    return safe_to_text(obj[key])
                            if "action" in obj and isinstance(obj["action"], dict):
                                return safe_to_text(obj["action"].get("content"))
                        try:
                            txt = getattr(obj, "text", None)
                            if isinstance(txt, str):
                                return txt
                        except Exception:
                            pass
                        try:
                            return json.dumps(obj, ensure_ascii=False, default=str)
                        except Exception:
                            return str(obj)
                    
                    def get_role(agent_obj) -> str:
                        try:
                            if hasattr(agent_obj, "_persona") and isinstance(agent_obj._persona, dict):
                                return agent_obj._persona.get("role") or getattr(agent_obj, "role", "Unknown")
                        except Exception:
                            pass
                        return getattr(agent_obj, "role", "Unknown") or "Unknown"
                    
                    responses = []
                    
                    if st.session_state["interaction_mode"] == "group":
                        # Broadcast to all agents
                        world.broadcast(user_question)
                        
                        # Get response from each agent
                        for agent in st.session_state["active_agents"]:
                            # Temporarily disable memory consolidation
                            original_consolidate = getattr(agent, "consolidate_episode_memories", None)
                            if callable(original_consolidate):
                                try:
                                    agent.consolidate_episode_memories = lambda: None
                                except Exception:
                                    original_consolidate = None
                            
                            try:
                                msg = agent.listen_and_act(
                                    "Respond to the follow-up question concisely.",
                                    return_actions=True
                                )
                                text = safe_to_text(msg).strip()
                                if text:
                                    responses.append({
                                        "speaker": agent.name,
                                        "role": get_role(agent),
                                        "text": text,
                                        "mode": "group"
                                    })
                            except Exception as e:
                                st.error(f"Error getting response from {agent.name}: {str(e)}")
                            finally:
                                if callable(original_consolidate):
                                    try:
                                        agent.consolidate_episode_memories = original_consolidate
                                    except Exception:
                                        pass
                            
                            time.sleep(pacing_seconds)
                        
                        st.session_state["total_api_calls"] += len(st.session_state["active_agents"])
                    
                    else:  # individual mode
                        if target_agent:
                            # Send message to specific agent
                            original_consolidate = getattr(target_agent, "consolidate_episode_memories", None)
                            if callable(original_consolidate):
                                try:
                                    target_agent.consolidate_episode_memories = lambda: None
                                except Exception:
                                    original_consolidate = None
                            
                            try:
                                target_agent.listen(user_question)
                                msg = target_agent.act(return_actions=True)
                                text = safe_to_text(msg).strip()
                                if text:
                                    responses.append({
                                        "speaker": target_agent.name,
                                        "role": get_role(target_agent),
                                        "text": text,
                                        "mode": "individual"
                                    })
                            except Exception as e:
                                st.error(f"Error getting response: {str(e)}")
                            finally:
                                if callable(original_consolidate):
                                    try:
                                        target_agent.consolidate_episode_memories = original_consolidate
                                    except Exception:
                                        pass
                            
                            st.session_state["total_api_calls"] += 1
                    
                    # Store interaction
                    if responses:
                        st.session_state["interaction_history"].append({
                            "user_question": user_question,
                            "responses": responses,
                            "timestamp": time.time()
                        })
                        st.rerun()
        
        # Display interaction history
        if st.session_state["interaction_history"]:
            st.markdown("### Follow-Up Discussion")
            for idx, interaction in enumerate(st.session_state["interaction_history"]):
                with st.chat_message("user"):
                    st.markdown(f"**You:** {interaction['user_question']}")
                
                for resp in interaction["responses"]:
                    with st.chat_message(resp["speaker"]):
                        mode_badge = "👥" if resp["mode"] == "group" else "👤"
                        st.markdown(f"{mode_badge} **({resp['role']})** {resp['text']}")
            
            # Export interaction history
            if st.button("📥 Download Interaction History"):
                interaction_data = {
                    "original_simulation": {
                        "model": model,
                        "turns": turns,
                        "seed": seed,
                        "transcript": transcript
                    },
                    "interactions": st.session_state["interaction_history"]
                }
                interaction_json = json.dumps(interaction_data, ensure_ascii=False, indent=2)
                st.download_button(
                    "💾 Save as JSON",
                    data=interaction_json,
                    file_name="interaction_history.json",
                    mime="application/json"
                )
    else:
        st.info("Run a simulation first to enable interactive follow-up questions.")

# ---------- Footer ----------
st.markdown("---")
st.caption(f"Model: `{model}` • Temperature: {temperature} • Turns: {turns} • Personas: {len(selected_names) if selected_names else 0}")
