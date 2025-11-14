# app_updated.py
# Enhanced Streamlit app using the new consolidated simulation engine
# This version provides all the same features with better scalability and error handling

import streamlit as st
import json
import time
import copy
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Import the new simulation engine
from scripts.simulation_engine import SimulationEngine, SimulationConfig, PersonaSpec
from tinytroupe.openai_utils import force_api_cache

# ---------- App Setup ----------
st.set_page_config(page_title="TinyTroupe Persona Simulator — Production", layout="wide")

st.title("🧠 TinyTroupe Persona Simulator — Production")
st.markdown(
    """
    Welcome to the enhanced TinyTroupe Persona Simulator! This production-ready tool provides 
    scalable persona-based discussions with comprehensive monitoring, validation, and analysis.
    
    **Features:**
    - 🚀 **Scalable Architecture**: Built for high-volume simulations
    - 🔍 **Real-time Analysis**: Automatic conversation analytics
    - 🛡️ **Error Handling**: Graceful degradation and monitoring
    - 📊 **Performance Metrics**: Detailed performance tracking
    - 🎯 **Persona Validation**: Comprehensive persona verification
    
    **How to use:**
    1. Configure simulation parameters in the sidebar
    2. Add custom personas or use defaults
    3. Enter your discussion topic
    4. Run simulation and view real-time results
    """
)

# Initialize simulation engine with production config
@st.cache_resource
def initialize_engine():
    config = SimulationConfig(
        output_dir="outputs",
        auto_analysis=True,
        enable_monitoring=True,
        graceful_degradation=True,
        performance_tracking=True,
        parallel_execution=True,
        max_workers=3,
        rate_limit_delay=1.0
    )
    return SimulationEngine(config)

engine = initialize_engine()

DATA_DIR = Path("data")
PERSONAS_JSON = DATA_DIR / "personas.agents.json"

# ---------- Rate Limiting & Security ----------
if "run_count" not in st.session_state:
    st.session_state["run_count"] = 0
if "last_run_time" not in st.session_state:
    st.session_state["last_run_time"] = None
if "total_api_calls" not in st.session_state:
    st.session_state["total_api_calls"] = 0
if "custom_agents" not in st.session_state:
    st.session_state["custom_agents"] = []
if "simulation_results" not in st.session_state:
    st.session_state["simulation_results"] = None

# Security limits
MAX_RUNS_PER_HOUR = 15
COOLDOWN_SECONDS = 20
MAX_API_CALLS_PER_SESSION = 2000
MAX_AGENTS = 8
MAX_TURNS_LIMIT = 10

# ---------- Sidebar Controls ----------
st.sidebar.header("🎛️ Simulation Settings")

# Model & Performance
col1, col2 = st.sidebar.columns(2)
with col1:
    model = st.selectbox(
        "Model",
        options=["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o"],
        index=0,
        help="Choose the language model"
    )

with col2:
    temperature = st.slider(
        "Temperature", 0.0, 1.5, 0.7, 0.1,
        help="Creativity level"
    )

# Simulation parameters
col1, col2 = st.sidebar.columns(2)
with col1:
    turns = st.slider("Turns", 1, MAX_TURNS_LIMIT, 6, help="Conversation rounds")

with col2:
    max_personas = st.slider("Max Personas", 1, MAX_AGENTS, 5, help="Limit for performance")

# Advanced options
with st.sidebar.expander("⚙️ Advanced Options", expanded=False):
    parallel_execution = st.checkbox("Parallel Execution", value=True, help="Enable parallel processing")
    auto_analysis = st.checkbox("Auto Analysis", value=True, help="Generate automatic analysis")
    rate_limit_delay = st.slider("Rate Limit Delay", 0.5, 3.0, 1.0, 0.1, help="Delay between API calls")
    enable_caching = st.checkbox("Enable API Caching", value=True, help="Cache API calls for faster reruns")

# Load persona specs with validation
@st.cache_data
def load_persona_specs():
    """Load and validate persona specifications."""
    try:
        if not PERSONAS_JSON.exists():
            st.error(f"Personas file not found: {PERSONAS_JSON}")
            return []
        
        with PERSONAS_JSON.open('r', encoding='utf-8') as f:
            data = json.load(f)
        
        personas = []
        for spec in data:
            try:
                persona = PersonaSpec(
                    id=spec.get('id', ''),
                    name=spec.get('name', ''),
                    role=spec.get('role', ''),
                    description=spec.get('description', ''),
                    personality=spec.get('personality', {}),
                    model=spec.get('model', model),
                    temperature=spec.get('temperature', temperature)
                )
                
                # Validate persona
                issues = persona.validate()
                if not issues:
                    personas.append(persona)
                else:
                    st.warning(f"Persona {persona.name} has validation issues: {', '.join(issues)}")
            
            except Exception as e:
                st.warning(f"Failed to load persona: {e}")
        
        return personas
        
    except Exception as e:
        st.error(f"Failed to load personas: {e}")
        return []

persona_specs = load_persona_specs()
persona_names = [p.name for p in persona_specs]

# Persona selection
st.sidebar.subheader("👥 Select Personas")
selected_names = st.sidebar.multiselect(
    "Participating Personas",
    options=persona_names,
    default=persona_names[:max_personas],
    help=f"Select up to {max_personas} personas"
)

# Validate selection
if len(selected_names) > max_personas:
    st.sidebar.warning(f"Too many personas selected. Limited to {max_personas}.")
    selected_names = selected_names[:max_personas]

# Security & monitoring panel
with st.sidebar.expander("🔒 Security & Monitoring", expanded=False):
    st.metric("Runs This Session", st.session_state["run_count"])
    st.metric("API Calls Used", st.session_state["total_api_calls"])
    
    # Estimate costs
    est_tokens = len(selected_names) * turns * 500
    est_cost = (est_tokens / 1000) * 0.002
    st.metric("Est. Tokens", f"{est_tokens:,}")
    st.metric("Est. Cost", f"${est_cost:.4f}")
    
    # Warnings
    if st.session_state["total_api_calls"] > MAX_API_CALLS_PER_SESSION * 0.8:
        st.warning("⚠️ Approaching session limit")

# ---------- Custom Persona Creation ----------
st.sidebar.markdown("---")
with st.sidebar.expander("➕ Create Custom Persona", expanded=False):
    st.caption("Design a new persona for your simulation")
    
    with st.form("custom_persona_form"):
        ca_name = st.text_input("Name*", placeholder="e.g., Alex Johnson")
        ca_role = st.text_input("Role*", placeholder="e.g., Senior Developer")
        ca_description = st.text_area("Description*", placeholder="Brief background and expertise")
        
        col1, col2 = st.columns(2)
        with col1:
            ca_traits = st.text_input("Traits", placeholder="analytical,creative,pragmatic")
        with col2:
            ca_style = st.text_input("Style", placeholder="Concise and technical")
        
        ca_goals = st.text_area("Goals", placeholder="One goal per line")
        
        submitted = st.form_submit_button("Add Persona")
        
        if submitted and ca_name and ca_role and ca_description:
            # Create new persona
            new_persona = PersonaSpec(
                id=ca_name.lower().replace(' ', '_'),
                name=ca_name,
                role=ca_role,
                description=ca_description,
                personality={
                    'traits': [t.strip() for t in ca_traits.split(',') if t.strip()],
                    'goals': [g.strip() for g in ca_goals.splitlines() if g.strip()],
                    'style': ca_style or 'Professional and collaborative'
                },
                model=model,
                temperature=temperature
            )
            
            # Validate
            issues = new_persona.validate()
            if not issues:
                st.session_state["custom_agents"].append(new_persona)
                st.success(f"✓ Added {ca_name}")
                st.rerun()
            else:
                st.error(f"Validation failed: {', '.join(issues)}")

# Show custom personas
custom_personas = st.session_state.get("custom_agents", [])
if custom_personas:
    st.sidebar.subheader("Custom Personas")
    for i, persona in enumerate(custom_personas):
        col1, col2 = st.sidebar.columns([3, 1])
        with col1:
            st.caption(f"👤 {persona.name} ({persona.role})")
        with col2:
            if st.button("🗑️", key=f"delete_{i}", help="Remove persona"):
                st.session_state["custom_agents"].pop(i)
                st.rerun()

# ---------- Main Interface ----------

# Discussion topic
seed = st.text_area(
    "💭 Discussion Topic",
    value="Evaluate the proposal for an AI-powered content moderation system that can detect harmful content across multiple languages and cultural contexts. Consider technical feasibility, ethical implications, and user impact.",
    height=120,
    help="Describe the topic, feature, or scenario for discussion"
)

# Environment configuration
with st.expander("🌍 Environment Settings", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        world_context = st.text_area(
            "Discussion Context",
            value="Cross-functional product team meeting\nFocus on practical solutions",
            help="Set the context for the discussion"
        )
    
    with col2:
        discussion_goal = st.text_area(
            "Discussion Goal",
            value="Generate actionable insights\nConsider multiple perspectives\nReach practical conclusions",
            help="What should the discussion achieve?"
        )

# Run simulation
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    run_clicked = st.button("🚀 Run Simulation", type="primary", disabled=not selected_names or not seed.strip())

with col2:
    if st.button("🧹 Clear Results"):
        st.session_state["simulation_results"] = None
        st.rerun()

with col3:
    validate_clicked = st.button("✅ Validate Setup")

# Validation
if validate_clicked:
    with st.spinner("Validating setup..."):
        issues = []
        
        # Check personas
        if not selected_names:
            issues.append("No personas selected")
        
        # Check seed
        if not seed.strip():
            issues.append("No discussion topic provided")
        
        # Check limits
        if len(selected_names) > MAX_AGENTS:
            issues.append(f"Too many personas (max {MAX_AGENTS})")
        
        # Validate selected personas
        valid_personas = 0
        for name in selected_names:
            persona = next((p for p in persona_specs if p.name == name), None)
            if persona:
                persona_issues = persona.validate()
                if not persona_issues:
                    valid_personas += 1
                else:
                    issues.extend([f"{name}: {issue}" for issue in persona_issues])
        
        if issues:
            st.error("❌ Validation Issues:")
            for issue in issues:
                st.write(f"• {issue}")
        else:
            st.success(f"✅ Setup valid! Ready to simulate with {valid_personas} personas.")

# Run simulation logic
if run_clicked:
    # Security checks
    current_time = time.time()
    
    # Rate limiting
    if st.session_state["last_run_time"]:
        time_since_last = current_time - st.session_state["last_run_time"]
        if time_since_last < COOLDOWN_SECONDS:
            remaining = int(COOLDOWN_SECONDS - time_since_last)
            st.error(f"⚠️ Please wait {remaining} more seconds before running again.")
            st.stop()
    
    # Session limits
    if st.session_state["run_count"] >= MAX_RUNS_PER_HOUR:
        st.error(f"⚠️ Session limit reached ({MAX_RUNS_PER_HOUR} runs). Please refresh to reset.")
        st.stop()
    
    if st.session_state["total_api_calls"] >= MAX_API_CALLS_PER_SESSION:
        st.error(f"⚠️ API call limit reached. Please refresh to reset.")
        st.stop()
    
    # Final validation
    if not selected_names:
        st.error("⚠️ Please select at least one persona.")
        st.stop()
    
    if not seed.strip():
        st.error("⚠️ Please enter a discussion topic.")
        st.stop()
    
    # Configure engine for this run
    run_config = SimulationConfig(
        model=model,
        temperature=temperature,
        max_turns=turns,
        output_dir="outputs",
        auto_analysis=auto_analysis,
        enable_monitoring=True,
        graceful_degradation=True,
        parallel_execution=parallel_execution,
        rate_limit_delay=rate_limit_delay,
        max_personas=max_personas
    )
    
    # Create engine for this run
    run_engine = SimulationEngine(run_config)
    
    # Prepare personas
    selected_personas = []
    
    # Add regular personas
    for name in selected_names:
        persona = next((p for p in persona_specs if p.name == name), None)
        if persona:
            selected_personas.append(persona)
    
    # Add custom personas
    for custom_persona in custom_personas:
        if custom_persona.name in selected_names:
            selected_personas.append(custom_persona)
    
    # Apply caching
    force_api_cache(enable_caching)
    
    # Run simulation
    with st.spinner("🔄 Running simulation..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Update session tracking
            st.session_state["last_run_time"] = current_time
            st.session_state["run_count"] += 1
            
            status_text.text("Initializing simulation...")
            progress_bar.progress(0.1)
            
            # Run the simulation
            results = run_engine.run_simulation(
                seed=seed.strip(),
                turns=turns,
                custom_personas=selected_personas
            )
            
            progress_bar.progress(0.9)
            status_text.text("Processing results...")
            
            if results.get('success', False):
                # Update API call tracking
                st.session_state["total_api_calls"] += results['metrics']['api_calls']
                
                # Store results
                st.session_state["simulation_results"] = results
                
                progress_bar.progress(1.0)
                status_text.text("✅ Simulation complete!")
                
                # Clear progress after a moment
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()
                
            else:
                st.error(f"❌ Simulation failed: {results.get('error', 'Unknown error')}")
                progress_bar.empty()
                status_text.empty()
        
        except Exception as e:
            st.error(f"❌ Simulation error: {str(e)}")
            progress_bar.empty()
            status_text.empty()

# ---------- Display Results ----------
results = st.session_state.get("simulation_results")

if results and results.get('success', False):
    st.success(f"🎉 Simulation completed! Generated {len(results['transcript'])} messages in {results['metrics']['duration']:.1f}s")
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Duration", f"{results['metrics']['duration']:.1f}s")
    
    with col2:
        st.metric("Messages", results['metrics']['total_messages'])
    
    with col3:
        st.metric("API Calls", results['metrics']['api_calls'])
    
    with col4:
        st.metric("Avg Response", f"{results['metrics']['average_response_time']:.2f}s")
    
    # Analysis tab
    if auto_analysis:
        with st.expander("📊 Analysis Dashboard", expanded=True):
            # Generate analysis
            analysis = engine.analyze_conversation(results['transcript'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Participation")
                
                # Create participation chart data
                participation_data = []
                for speaker, stats in analysis['participation']['by_speaker'].items():
                    participation_data.append({
                        'Speaker': speaker,
                        'Messages': stats['messages'],
                        'Words': stats['words'],
                        'Avg Length': stats['words'] / stats['messages'] if stats['messages'] > 0 else 0
                    })
                
                st.dataframe(participation_data, hide_index=True)
            
            with col2:
                st.subheader("Content Focus")
                
                focus_data = analysis['content_analysis']['focus_areas']
                total_focus = sum(focus_data.values())
                
                if total_focus > 0:
                    for area, count in focus_data.items():
                        percentage = (count / total_focus) * 100
                        st.write(f"**{area.title()}**: {count} mentions ({percentage:.1f}%)")
                        st.progress(percentage / 100)
                else:
                    st.write("No specific focus areas detected")
            
            # Interaction patterns
            st.subheader("Interaction Patterns")
            
            top_patterns = analysis['interaction_patterns']['most_active_pairs'][:5]
            if top_patterns:
                pattern_data = [{'Pattern': pattern, 'Frequency': freq} for pattern, freq in top_patterns]
                st.dataframe(pattern_data, hide_index=True)
            else:
                st.write("No clear interaction patterns detected")
    
    # Transcript display
    st.subheader("📝 Conversation Transcript")
    
    # Filter options
    col1, col2 = st.columns([1, 3])
    with col1:
        speaker_filter = st.selectbox(
            "Filter by speaker",
            options=["All"] + list(set(msg['speaker'] for msg in results['transcript'])),
            index=0
        )
    
    # Display messages
    filtered_transcript = results['transcript']
    if speaker_filter != "All":
        filtered_transcript = [msg for msg in results['transcript'] if msg['speaker'] == speaker_filter]
    
    for msg in filtered_transcript:
        with st.chat_message(msg['speaker']):
            st.markdown(f"**Turn {msg['turn']} - {msg['role']}**")
            st.write(msg['text'])
    
    # Export options
    st.subheader("💾 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # JSON export
        json_data = json.dumps(results, indent=2, ensure_ascii=False, default=str)
        st.download_button(
            "📊 Download JSON",
            data=json_data,
            file_name=f"simulation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with col2:
        # JSONL export
        jsonl_data = "\n".join(json.dumps(msg, ensure_ascii=False) for msg in results['transcript'])
        st.download_button(
            "📋 Download JSONL",
            data=jsonl_data,
            file_name=f"conversation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
            mime="application/json"
        )
    
    with col3:
        # Markdown report
        if auto_analysis:
            # Generate markdown report
            analysis = engine.analyze_conversation(results['transcript'])
            
            md_lines = [
                f"# Simulation Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                "",
                "## Configuration",
                f"- **Model**: {results['config']['model']}",
                f"- **Temperature**: {results['config']['temperature']}",
                f"- **Turns**: {results['config']['turns']}",
                f"- **Topic**: {results['config']['seed']}",
                "",
                "## Performance",
                f"- **Duration**: {results['metrics']['duration']:.1f}s",
                f"- **Messages**: {results['metrics']['total_messages']}",
                f"- **API Calls**: {results['metrics']['api_calls']}",
                "",
                "## Participants",
            ]
            
            for agent in results['agents']:
                md_lines.append(f"- **{agent['name']}** ({agent['role']})")
            
            md_lines.extend([
                "",
                "## Analysis",
                f"- **Dominant Focus**: {analysis['content_analysis']['dominant_focus'].title()}",
                f"- **Unique Speakers**: {analysis['overview']['unique_speakers']}",
                f"- **Avg Message Length**: {analysis['overview']['avg_message_length']:.1f} words",
                "",
                "## Transcript",
                ""
            ])
            
            for msg in results['transcript']:
                md_lines.extend([
                    f"**Turn {msg['turn']} - {msg['speaker']} ({msg['role']})**",
                    msg['text'],
                    ""
                ])
            
            md_data = "\n".join(md_lines)
            
            st.download_button(
                "📝 Download Report",
                data=md_data,
                file_name=f"simulation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )

# ---------- Footer ----------
st.markdown("---")

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    if results:
        st.caption(f"✅ Last run: {results['metrics']['total_messages']} messages, {results['metrics']['duration']:.1f}s")
    else:
        st.caption("Ready for simulation")

with col2:
    st.caption(f"Session: {st.session_state['run_count']}/{MAX_RUNS_PER_HOUR} runs")

with col3:
    st.caption(f"API calls: {st.session_state['total_api_calls']:,}")

# Additional monitoring info
if st.checkbox("Show Debug Info", value=False):
    st.subheader("Debug Information")
    
    debug_info = {
        "Engine Config": {
            "Model": engine.config.model,
            "Temperature": engine.config.temperature,
            "Parallel Execution": engine.config.parallel_execution,
            "Rate Limit Delay": engine.config.rate_limit_delay,
            "Max Personas": engine.config.max_personas,
        },
        "Session State": {
            "Run Count": st.session_state["run_count"],
            "Total API Calls": st.session_state["total_api_calls"],
            "Custom Agents": len(st.session_state.get("custom_agents", [])),
            "Has Results": bool(st.session_state.get("simulation_results")),
        },
        "Current Selection": {
            "Selected Personas": len(selected_names),
            "Turns": turns,
            "Seed Length": len(seed.strip()),
        }
    }
    
    st.json(debug_info)