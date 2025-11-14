# app_enhanced.py
# Aesthetically Enhanced Streamlit app with modern design and custom styling
# Built on the consolidated simulation engine with AI-powered features

import streamlit as st
import json
import time
import copy
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# ========================================
# Hugging Face Secrets Configuration
# ========================================
def setup_api_key():
    """Configure OpenAI API key from Hugging Face secrets or environment."""
    # Priority 1: Check Streamlit secrets (Hugging Face)
    if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
        os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
        return True
    
    # Priority 2: Check environment variable
    if 'OPENAI_API_KEY' in os.environ and os.environ['OPENAI_API_KEY']:
        return True
    
    # Priority 3: Check config file
    config_path = Path('config/config.ini')
    if config_path.exists():
        try:
            import configparser
            parser = configparser.ConfigParser()
            parser.read(config_path)
            if 'OpenAI' in parser and 'api_key' in parser['OpenAI']:
                api_key = parser['OpenAI']['api_key']
                if api_key and not api_key.startswith('#') and api_key != 'sk-your-api-key-here':
                    os.environ['OPENAI_API_KEY'] = api_key
                    return True
        except Exception:
            pass
    
    return False

# Configure API key before importing simulation engine
has_api_key = setup_api_key()

# Import the simulation engine
from scripts.simulation_engine import SimulationEngine, SimulationConfig, PersonaSpec
from tinytroupe.openai_utils import force_api_cache

# ---------- Enhanced Page Configuration ----------
st.set_page_config(
    page_title="🧠 AI Persona Panel Simulator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/microsoft/tinytroupe',
        'Report a bug': None,
        'About': "AI-powered persona panel simulation using TinyTroupe with advanced cognitive features"
    }
)

# ---------- Custom CSS Styling ----------
st.markdown("""
<style>
/* Import modern fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Root variables for consistent theming */
:root {
    --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    --success-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    --warning-gradient: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    --glass-bg: rgba(255, 255, 255, 0.95);
    --glass-border: rgba(0, 0, 0, 0.1);
    --text-primary: #000000;
    --text-secondary: #333333;
    --shadow-soft: 0 8px 32px rgba(0, 0, 0, 0.12);
    --shadow-strong: 0 20px 60px rgba(0, 0, 0, 0.25);
}

/* Global styles */
* {
    font-family: 'Inter', sans-serif;
    color: #000000 !important;
}

.main .block-container {
    padding-top: 2rem;
    max-width: 1200px;
    background-color: #ffffff;
}

/* Streamlit specific overrides */
.stApp {
    background-color: #f8f9fa;
    color: #000000;
}

.stSelectbox label,
.stMultiSelect label,
.stSlider label,
.stTextArea label,
.stCheckbox label,
.stRadio label {
    color: #000000 !important;
}

.stMarkdown,
.stMarkdown p,
.stMarkdown div,
.stMarkdown span {
    color: #000000 !important;
}

.stSidebar {
    background-color: #ffffff;
}

.stSidebar .stMarkdown,
.stSidebar .stSelectbox,
.stSidebar .stSlider,
.stSidebar .stCheckbox {
    color: #000000 !important;
}

/* Header styling */
.hero-header {
    background: var(--primary-gradient);
    padding: 3rem 2rem;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 3rem;
    box-shadow: var(--shadow-strong);
    backdrop-filter: blur(20px);
    border: 1px solid var(--glass-border);
    position: relative;
    overflow: hidden;
}

.hero-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="20" cy="20" r="2" fill="rgba(255,255,255,0.1)"/><circle cx="80" cy="30" r="1.5" fill="rgba(255,255,255,0.08)"/><circle cx="40" cy="70" r="1" fill="rgba(255,255,255,0.06)"/><circle cx="90" cy="80" r="2.5" fill="rgba(255,255,255,0.04)"/></svg>') repeat;
    animation: float 20s linear infinite;
}

@keyframes float {
    0% { transform: translateY(0px) rotate(0deg); }
    100% { transform: translateY(-100px) rotate(360deg); }
}

.hero-title {
    color: white;
    font-size: 3.5rem;
    font-weight: 700;
    margin: 0;
    text-shadow: 2px 2px 20px rgba(0, 0, 0, 0.4);
    letter-spacing: -1px;
}

.hero-subtitle {
    color: rgba(255, 255, 255, 0.95);
    font-size: 1.3rem;
    margin: 1rem 0;
    font-weight: 400;
}

.hero-description {
    color: rgba(255, 255, 255, 0.8);
    font-size: 1.1rem;
    margin: 0;
    font-weight: 300;
}

/* Glass morphism cards */
.glass-card {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(0, 0, 0, 0.1);
    border-radius: 16px;
    padding: 2rem;
    margin: 1.5rem 0;
    box-shadow: var(--shadow-soft);
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    position: relative;
    overflow: hidden;
    color: #000000 !important;
}

.glass-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,0,0,0.2), transparent);
}

.glass-card:hover {
    transform: translateY(-8px);
    box-shadow: var(--shadow-strong);
    border-color: rgba(0, 0, 0, 0.2);
}

.glass-card h2, .glass-card h3 {
    color: #000000 !important;
    margin-top: 0;
}

.glass-card p {
    color: #333333 !important;
}

/* Sidebar enhancements */
.css-1d391kg {
    background: #ffffff;
    color: #000000 !important;
}

.sidebar-header {
    background: var(--primary-gradient);
    padding: 1.5rem;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow-soft);
}

.sidebar-header h3 {
    color: white !important;
    margin: 0;
    font-weight: 600;
}

/* Status indicators */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    border-radius: 25px;
    font-size: 0.9rem;
    font-weight: 500;
    margin: 0.5rem;
    box-shadow: var(--shadow-soft);
    transition: all 0.3s ease;
}

.status-active {
    background: var(--success-gradient);
    color: white;
}

.status-processing {
    background: var(--warning-gradient);
    color: white;
    animation: pulse 2s infinite;
}

.status-ai-powered {
    background: var(--primary-gradient);
    color: white;
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.8; transform: scale(1.05); }
}

/* Metric cards */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1.5rem;
    margin: 2rem 0;
}

.metric-card {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(0, 0, 0, 0.1);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: var(--primary-gradient);
}

.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: var(--shadow-strong);
}

.metric-value {
    font-size: 2.5rem;
    font-weight: 700;
    color: #000000 !important;
    margin: 0.5rem 0;
}

.metric-label {
    font-size: 0.9rem;
    color: #333333 !important;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.metric-description {
    font-size: 0.8rem;
    color: #666666 !important;
    margin-top: 0.5rem;
    opacity: 0.8;
}

/* Enhanced buttons */
.stButton > button {
    background: var(--primary-gradient);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.8rem 2rem;
    font-weight: 600;
    font-size: 1.1rem;
    transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    box-shadow: var(--shadow-soft);
    text-transform: none;
    letter-spacing: 0.3px;
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: var(--shadow-strong);
    background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%);
}

.stButton > button:active {
    transform: translateY(0);
}

/* Form elements */
.stSelectbox > div > div,
.stMultiSelect > div > div,
.stSlider > div > div {
    background: rgba(255, 255, 255, 0.95);
    border: 1px solid rgba(0, 0, 0, 0.1);
    border-radius: 8px;
    backdrop-filter: blur(10px);
    color: #000000 !important;
}

.stTextArea > div > div > textarea {
    background: rgba(255, 255, 255, 0.95) !important;
    border: 1px solid rgba(0, 0, 0, 0.1) !important;
    border-radius: 12px !important;
    color: #000000 !important;
    backdrop-filter: blur(10px);
    font-family: 'Inter', sans-serif;
}

.stTextArea > div > div > textarea::placeholder {
    color: #666666 !important;
    opacity: 0.8;
}

/* Loading animation */
.loading-spinner {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid rgba(255,255,255,0.3);
    border-radius: 50%;
    border-top-color: #fff;
    animation: spin 1s ease-in-out infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

/* Feature highlights */
.feature-highlight {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
    border: 1px solid rgba(102, 126, 234, 0.3);
    border-radius: 12px;
    padding: 1rem;
    margin: 1rem 0;
    backdrop-filter: blur(10px);
    color: #000000 !important;
}

.feature-highlight strong {
    color: #667eea !important;
}

/* Topic input text area styling */
div[data-testid="stTextArea"] textarea {
    color: #000000 !important;
    background-color: #ffffff !important;
    border: 2px solid #e2e8f0 !important;
    border-radius: 8px !important;
}

/* Alternative selector for topic text area */
.stTextArea > div > div > textarea {
    color: #000000 !important;
    background-color: #ffffff !important;
}

/* Force all text areas to have black text */
textarea {
    color: #000000 !important;
    background-color: #ffffff !important;
}

/* Responsive design */
@media (max-width: 768px) {
    .hero-title {
        font-size: 2.5rem;
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
    }
    
    .metric-grid {
        grid-template-columns: 1fr;
    }
}
</style>

<script>
// Force black text in text areas after page load
setTimeout(function() {
    const textAreas = document.querySelectorAll('textarea');
    textAreas.forEach(function(textarea) {
        textarea.style.color = '#000000';
        textarea.style.backgroundColor = '#ffffff';
        textarea.style.setProperty('color', '#000000', 'important');
        textarea.style.setProperty('background-color', '#ffffff', 'important');
    });
}, 1000);

// Also try on input events
document.addEventListener('DOMContentLoaded', function() {
    setTimeout(function() {
        const textAreas = document.querySelectorAll('textarea');
        textAreas.forEach(function(textarea) {
            textarea.style.setProperty('color', '#000000', 'important');
            textarea.style.setProperty('background-color', '#ffffff', 'important');
        });
    }, 2000);
});
</script>
""", unsafe_allow_html=True)

# ---------- Hero Header ----------
st.markdown("""
<div class="hero-header">
    <h1 class="hero-title">🧠 AI Persona Panel Simulator</h1>
    <p class="hero-subtitle">Powered by Microsoft's TinyTroupe Framework</p>
    <p class="hero-description">✨ Enhanced with Session Memory & Intelligent Agent Selection ✨</p>
</div>
""", unsafe_allow_html=True)

# ---------- API Key Check ----------
if not has_api_key:
    st.error("⚠️ **OpenAI API Key Required**")
    st.markdown("""
    This application requires an OpenAI API key to function. Please configure it using one of these methods:
    
    **For Hugging Face Spaces:**
    1. Go to your Space **Settings** tab
    2. Scroll to **Repository secrets** section
    3. Click **New secret**
    4. Add a secret with:
       - **Name**: `OPENAI_API_KEY`
       - **Value**: Your OpenAI API key (starts with `sk-`)
    5. Click **Add secret**
    6. The Space will restart automatically
    
    **For Local Deployment:**
    - Edit `config/config.ini` and add your API key under `[OpenAI]` section
    - Or set the `OPENAI_API_KEY` environment variable
    
    **Get an API Key:** Visit https://platform.openai.com/api-keys
    """)
    st.stop()

# ---------- Initialize Engine ----------
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
        enable_smart_agent_selection=True,
        enable_session_memory=True,
        enable_recall_faculty=True,
        enable_collaborative_tools=True
    )
    return SimulationEngine(config)

try:
    simulation_engine = initialize_engine()
    
    # Force load personas directly
    persona_specs = simulation_engine.persona_db.load_personas()
    personas_dict = simulation_engine.persona_db.get_all_personas()
    persona_names = list(personas_dict.keys())
    
    # Add custom personas from session state
    if 'custom_personas' in st.session_state:
        custom_names = [p.name for p in st.session_state.custom_personas]
        persona_names.extend(custom_names)
    
    personas_loaded = True
    

except Exception as e:
    st.error(f"Failed to initialize simulation engine: {e}")
    st.error(f"Error details: {str(e)}")
    import traceback
    st.error(f"Traceback: {traceback.format_exc()}")
    personas_loaded = False
    persona_names = []

# ---------- Enhanced Sidebar ----------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <h3>⚙️ Configuration Panel</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Configuration (Collapsible)
    with st.expander("🤖 AI Model Configuration", expanded=True):
        model = st.selectbox(
            "Model",
            ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            help="Choose the AI model for generating responses"
        )
        temperature = st.slider("🌡️ Creativity", 0.0, 1.0, 0.7, 0.1)
    
    # Environment & Context Settings (New Section)
    with st.expander("🌍 Environment & Context", expanded=False):
        environment_name = st.text_input(
            "Environment Name", 
            value="DiscussionRoom",
            help="Name of the TinyTroupe environment for the simulation"
        )
        
        environment_description = st.text_area(
            "Environment Description",
            value="A professional meeting room designed for collaborative discussions and brainstorming sessions.",
            height=80,
            help="Description of the environment setting"
        )
        
        custom_context = st.text_area(
            "Additional Context",
            value="",
            placeholder="Add any additional context or ground rules for the discussion...",
            height=100,
            help="Extra context that will be shared with all participants"
        )
        
        discussion_goal = st.selectbox(
            "Discussion Goal",
            [
                "Participate constructively in the discussion. Be concrete and practical.",
                "Focus on innovative solutions and creative thinking.",
                "Analyze pros and cons systematically.",
                "Prioritize user experience and practical implementation.",
                "Consider ethical implications and long-term consequences.",
                "Custom Goal"
            ]
        )
        
        if discussion_goal == "Custom Goal":
            custom_goal = st.text_input(
                "Custom Discussion Goal",
                placeholder="Enter your custom discussion goal..."
            )
        else:
            custom_goal = discussion_goal
    
    # Simulation Parameters (Collapsible)
    with st.expander("🎯 Simulation Parameters", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            turns = st.slider("🔄 Rounds", 3, 15, 6, 1)
        with col2:
            max_personas = st.slider("👥 Max Personas", 3, 10, 6, 1)
        
        rate_limit_delay = st.slider("⏱️ Rate Limit (seconds)", 0.5, 3.0, 1.0, 0.1)
    
    # Advanced Options (Collapsible)
    with st.expander("⚙️ Advanced Options", expanded=False):
        parallel_execution = st.checkbox("Parallel Processing", value=True)
        auto_analysis = st.checkbox("Auto Analysis", value=True)
        enable_caching = st.checkbox("API Caching", value=True)
        
        st.markdown("**Performance Settings**")
        col1, col2 = st.columns(2)
        with col1:
            max_workers = st.slider("Max Workers", 1, 5, 3)
        with col2:
            timeout_seconds = st.slider("Timeout (sec)", 60, 600, 300)
    
    # AI Enhancement Features (Collapsible)
    with st.expander("🧠 AI Enhancement Features", expanded=False):
        smart_agent_selection = st.checkbox("🎯 Smart Agent Selection", value=True)
        enable_session_memory = st.checkbox("💭 Session Memory", value=True)
        enable_recall_faculty = st.checkbox("🔍 Memory Recall", value=True)
        enable_collaborative_tools = st.checkbox("🛠️ Collaborative Tools", value=True)
    
    # Persona Selection (Collapsible)
    with st.expander("👥 Persona Selection", expanded=True):
        # Show available personas
        if personas_loaded and len(persona_names) > 0:
            st.caption(f"Available personas: {len(persona_names)}")
        else:
            st.error("❌ No personas available")
        
        if smart_agent_selection:
            st.markdown("""
            <div class="status-badge status-ai-powered">🤖 Persona Auto-Selection (Default) </div>
            """, unsafe_allow_html=True)
            manual_override = st.checkbox("Manual Persona Selection")
            if manual_override:
                if persona_names:
                    selected_names = st.multiselect("Select Specific Personas", persona_names, [], key="manual_persona_select")
                else:
                    st.warning("No personas available for selection")
                    selected_names = []
            else:
                selected_names = []
        else:
            if persona_names:
                selected_names = st.multiselect(
                    "Participating Personas",
                    persona_names,
                    persona_names[:max_personas] if len(persona_names) >= max_personas else persona_names,
                    key="standard_persona_select"
                )
            else:
                st.warning("No personas available for selection")
                selected_names = []
        
        # Validation
        if not smart_agent_selection and len(selected_names) > max_personas:
            st.warning(f"Limited to {max_personas} personas")
            selected_names = selected_names[:max_personas]
    
    # Custom Persona Creation
    st.markdown("""
    <div class="glass-card" style="padding: 1rem; margin: 1rem 0;">
        <h4>✨ Create Custom Persona</h4>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("🎭 Persona Creator", expanded=False):
        st.markdown("**Create a temporary persona for this session**")
        
        # Basic info
        col1, col2 = st.columns(2)
        with col1:
            custom_name = st.text_input("Persona Name", placeholder="Dr. Alex Smith", key="creator_name")
            custom_role = st.selectbox("Professional Role", [
                "Data Scientist", "Software Engineer", "Product Manager", "UX Designer", 
                "Marketing Expert", "Business Analyst", "Security Researcher", "Academic Researcher",
                "Consultant", "Entrepreneur", "Policy Maker", "Ethics Specialist"
            ], key="creator_role")
        
        with col2:
            custom_expertise = st.multiselect("Expertise Areas", [
                "Machine Learning", "Data Analytics", "Software Development", "Cloud Computing",
                "Cybersecurity", "Product Strategy", "User Experience", "Digital Marketing",
                "Business Strategy", "Project Management", "Research & Development",
                "Financial Analysis", "Risk Management", "Ethics & Compliance"
            ], key="creator_expertise")
            custom_industry = st.selectbox("Industry Background", [
                "Technology", "Healthcare", "Finance", "Education", "Government",
                "Retail", "Manufacturing", "Consulting", "Startup", "Non-profit"
            ], key="creator_industry")
        
        # Personality traits
        col3, col4 = st.columns(2)
        with col3:
            communication_style = st.selectbox("Communication Style", [
                "Direct", "Diplomatic", "Analytical", "Creative", "Supportive"
            ], key="creator_comm_style")
            thinking_approach = st.selectbox("Thinking Approach", [
                "Strategic", "Detail-oriented", "Big-picture", "Problem-solving", "Innovation-focused"
            ], key="creator_thinking")
        
        with col4:
            experience_level = st.selectbox("Experience Level", [
                "Junior", "Mid-level", "Senior", "Executive", "Expert"
            ], key="creator_experience")
            
            perspective = st.selectbox("Primary Perspective", [
                "Technical feasibility", "Business value", "User experience", "Risk assessment", 
                "Innovation potential", "Ethical considerations", "Cost-effectiveness"
            ], key="creator_perspective")
        
        # Advanced settings
        with st.expander("Advanced Settings", expanded=False):
            custom_temperature = st.slider("Response Creativity", 0.0, 1.0, 0.7, 0.1, key="creator_temp")
            response_style = st.selectbox("Response Style", [
                "Concise", "Detailed", "Balanced", "Question-focused", "Solution-oriented"
            ], key="creator_response_style")
        
        # Create button
        if st.button("🎭 Create Custom Persona", type="primary", key="create_persona_btn"):
            if custom_name.strip():
                from scripts.simulation_engine import PersonaSpec
                import time
                
                # Create persona data
                persona_dict = {
                    "id": f"custom_{int(time.time())}",
                    "name": custom_name.strip(),
                    "role": custom_role,
                    "description": f"A {experience_level.lower()} {custom_role.lower()} with expertise in {', '.join(custom_expertise[:3]) if custom_expertise else 'general business'}. Known for {communication_style.lower()} communication and {thinking_approach.lower()} approach. Focuses on {perspective.lower()}.",
                    "personality": {
                        "traits": [communication_style, thinking_approach, experience_level],
                        "goals": [f"Provide {perspective.lower()} insights", "Contribute valuable expertise"],
                        "style": f"{communication_style} and {response_style}",
                        "expertise": custom_expertise,
                        "industry": custom_industry,
                        "background": f"{experience_level} {custom_role} with {custom_industry} industry background"
                    },
                    "temperature": custom_temperature,
                    "prompting": {
                        "style_instructions": f"Respond as a {communication_style.lower()} {custom_role.lower()} with {thinking_approach.lower()} approach. Focus on {perspective.lower()}. Use a {response_style.lower()} response style."
                    }
                }
                
                # Validate persona before creating
                try:
                    from scripts.persona_validator import PersonaValidator
                    validator = PersonaValidator()
                    validation_result = validator.validate_persona_structure(persona_dict)
                    
                    # Show validation results
                    if not validation_result.is_valid:
                        st.error("❌ Persona validation failed:")
                        for issue in validation_result.issues:
                            st.error(f"  • {issue}")
                    else:
                        # Show validation scores
                        col_v1, col_v2 = st.columns(2)
                        with col_v1:
                            score_color = "🟢" if validation_result.consistency_score >= 0.8 else "🟡" if validation_result.consistency_score >= 0.6 else "🔴"
                            st.info(f"{score_color} Consistency: {validation_result.consistency_score:.0%}")
                        with col_v2:
                            score_color = "🟢" if validation_result.realism_score >= 0.8 else "🟡" if validation_result.realism_score >= 0.6 else "🔴"
                            st.info(f"{score_color} Realism: {validation_result.realism_score:.0%}")
                        
                        # Show warnings if any
                        if validation_result.warnings:
                            with st.expander("⚠️ Validation Warnings", expanded=False):
                                for warning in validation_result.warnings:
                                    st.warning(warning)
                        
                        # Show suggestions if any
                        if validation_result.suggestions:
                            with st.expander("💡 Improvement Suggestions", expanded=False):
                                for suggestion in validation_result.suggestions:
                                    st.info(suggestion)
                        
                        # Create PersonaSpec object
                        custom_persona = PersonaSpec(**persona_dict)
                        
                        # Store in session state
                        if 'custom_personas' not in st.session_state:
                            st.session_state.custom_personas = []
                        
                        # Check if persona with this name already exists
                        existing_names = [p.name for p in st.session_state.custom_personas]
                        if custom_name.strip() in existing_names:
                            st.warning(f"Persona '{custom_name.strip()}' already exists. Choose a different name.")
                        else:
                            st.session_state.custom_personas.append(custom_persona)
                            st.success(f"✨ Created custom persona: {custom_name.strip()}")
                            st.rerun()
                            
                except ImportError:
                    # Fallback if validator not available
                    custom_persona = PersonaSpec(**persona_dict)
                    
                    if 'custom_personas' not in st.session_state:
                        st.session_state.custom_personas = []
                    
                    existing_names = [p.name for p in st.session_state.custom_personas]
                    if custom_name.strip() in existing_names:
                        st.warning(f"Persona '{custom_name.strip()}' already exists. Choose a different name.")
                    else:
                        st.session_state.custom_personas.append(custom_persona)
                        st.success(f"✨ Created custom persona: {custom_name.strip()}")
                        st.rerun()
            else:
                st.warning("Please enter a persona name")
        
        # Show existing custom personas
        if 'custom_personas' in st.session_state and st.session_state.custom_personas:
            st.markdown("**Custom Personas Created This Session:**")
            for i, persona in enumerate(st.session_state.custom_personas):
                col_name, col_remove = st.columns([3, 1])
                with col_name:
                    st.caption(f"🎭 {persona.name} ({persona.role})")
                with col_remove:
                    if st.button("🗑️", key=f"remove_persona_{i}", help=f"Remove {persona.name}"):
                        st.session_state.custom_personas.pop(i)
                        st.rerun()

# ---------- Main Dashboard ----------
if not personas_loaded:
    st.markdown("""
    <div class="glass-card" style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);">
        <h3>❌ Configuration Error</h3>
        <p>Personas not loaded. Please check the personas file.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Dashboard metrics
st.markdown("""
<div class="glass-card">
    <h2>📊 Simulation Dashboard</h2>
    <p>Real-time overview of your persona panel configuration</p>
</div>
""", unsafe_allow_html=True)

# Calculate metrics
selected_count = len(selected_names) if selected_names else max_personas
base_tokens = selected_count * turns * 500
memory_overhead = 1.2 if enable_session_memory else 1.0
smart_selection_overhead = 1.1 if smart_agent_selection else 1.0
est_tokens = int(base_tokens * memory_overhead * smart_selection_overhead)
est_cost = (est_tokens / 1000) * 0.002
active_features = sum([smart_agent_selection, enable_session_memory, enable_recall_faculty, enable_collaborative_tools])

# Metrics grid
st.markdown(f"""
<div class="metric-grid">
    <div class="metric-card">
        <div class="metric-value">{len(persona_names)}</div>
        <div class="metric-label">👨‍🎓 Available Personas</div>
        <div class="metric-description">Ready for discussion</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{selected_count}</div>
        <div class="metric-label">✅ Selected Personas</div>
        <div class="metric-description">Will participate</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">${est_cost:.4f}</div>
        <div class="metric-label">💰 Estimated Cost</div>
        <div class="metric-description">{est_tokens:,} tokens</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{active_features}/4</div>
        <div class="metric-label">🧠 AI Features</div>
        <div class="metric-description">Enhanced capabilities</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------- Discussion Topic ----------
st.markdown("""
<div class="glass-card">
    <h2>💭 Discussion Topic</h2>
    <p>Describe your topic below - our AI personas will dive deep into the discussion!</p>
</div>
""", unsafe_allow_html=True)

seed = st.text_area(
    "",
    value="Evaluate the proposal for an AI-powered content moderation system that can detect harmful content across multiple languages and cultural contexts. Consider technical feasibility, ethical implications, and user impact.",
    height=120,
    help="The more specific your topic, the better our AI can select relevant personas!",
    placeholder="Enter your discussion topic here...",
    key="topic_input"
)

# ---------- Enhanced Simulation Button ----------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🚀 Launch AI Persona Panel Discussion", type="primary", use_container_width=True):
        if not seed.strip():
            st.error("Please provide a discussion topic!")
            st.stop()
        
        # Show loading state
        with st.spinner("Initializing AI Persona Panel..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Update progress
                progress_bar.progress(20)
                status_text.text("🔧 Configuring simulation engine...")
                
                # Configure engine
                config = SimulationConfig(
                    model=model,
                    temperature=temperature,
                    max_turns=turns,
                    output_dir="outputs",
                    auto_analysis=auto_analysis,
                    enable_monitoring=True,
                    graceful_degradation=True,
                    parallel_execution=parallel_execution,
                    rate_limit_delay=rate_limit_delay,
                    max_personas=max_personas,
                    enable_smart_agent_selection=smart_agent_selection,
                    enable_session_memory=enable_session_memory,
                    enable_recall_faculty=enable_recall_faculty,
                    enable_collaborative_tools=enable_collaborative_tools,
                    max_workers=max_workers,
                    timeout_seconds=timeout_seconds
                )
                
                run_engine = SimulationEngine(config)
                
                # Progress update
                progress_bar.progress(40)
                status_text.text("🤖 Preparing AI personas...")
                
                # Prepare personas
                selected_personas = []
                if selected_names:
                    # Get standard personas
                    all_personas = run_engine.persona_db.get_all_personas()
                    selected_personas = [all_personas[name] for name in selected_names if name in all_personas]
                    
                    # Add custom personas from session state
                    if 'custom_personas' in st.session_state:
                        custom_personas_dict = {p.name: p for p in st.session_state.custom_personas}
                        for name in selected_names:
                            if name in custom_personas_dict:
                                selected_personas.append(custom_personas_dict[name])
                
                # Progress update
                progress_bar.progress(60)
                status_text.text("💭 Starting persona discussion...")
                
                # Run simulation
                if smart_agent_selection and not selected_names:
                    results = run_engine.run_simulation(
                        seed=seed.strip(),
                        turns=turns,
                        custom_personas=selected_personas,
                        use_smart_selection=True,
                        environment_name=environment_name,
                        environment_description=environment_description,
                        custom_context=custom_context,
                        discussion_goal=custom_goal
                    )
                else:
                    results = run_engine.run_simulation(
                        seed=seed.strip(),
                        turns=turns,
                        custom_personas=selected_personas,
                        use_smart_selection=False,
                        environment_name=environment_name,
                        environment_description=environment_description,
                        custom_context=custom_context,
                        discussion_goal=custom_goal
                    )
                
                # Complete
                progress_bar.progress(100)
                status_text.text("✅ Discussion complete!")
                
                # Store agents in session state for reuse
                if results.get('success') and results.get('agent_objects'):
                    st.session_state['last_simulation_agents'] = results['agent_objects']
                    st.session_state['last_simulation_results'] = results
                    st.session_state['run_engine'] = run_engine  # Store engine for agent management
                
                # Reset environment name for next simulation
                if st.session_state.get('reset_environment', True):
                    st.session_state['environment_name_reset'] = True
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
            except Exception as e:
                st.error(f"❌ Simulation failed: {str(e)}")
                st.stop()
        
        # ---------- Display Results ----------
        if results and results.get('success', False):
            # Success header
            if results.get('selected_agents') and smart_agent_selection:
                st.markdown(f"""
                <div class="feature-highlight">
                    <strong>🤖 AI Selected Personas:</strong> {', '.join(results['selected_agents'])}
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="status-badge status-active" style="display: block; text-align: center; margin: 2rem 0;">
                🎉 Discussion completed! Generated {len(results['transcript'])} messages in {results['metrics']['duration']:.1f}s
            </div>
            """, unsafe_allow_html=True)
            
            # Session info
            if results.get('session_id'):
                st.caption(f"Session ID: {results['session_id']}")
            
            # Results tabs
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["💬 Conversation", "🧠 AI Thoughts", "💭 Agent Chat", "📊 Analysis", "📈 Metrics", "📁 Export"])
            
            with tab1:
                st.markdown("""
                <div class="glass-card">
                    <h3>💬 Persona Panel Discussion</h3>
                    <p>Final responses from each persona</p>
                </div>
                """, unsafe_allow_html=True)
                
                for i, msg in enumerate(results['transcript']):
                    speaker = msg.get('speaker', 'Unknown')
                    content = msg.get('content', msg.get('message', ''))
                    
                    with st.container():
                        st.markdown(f"""
                        <div class="glass-card" style="margin: 1rem 0; padding: 1rem;">
                            <strong style="color: #667eea;">💼 {speaker}</strong>
                            <p style="margin: 0.5rem 0 0 0; line-height: 1.6;">{content}</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            with tab2:
                st.markdown("""
                <div class="glass-card">
                    <h3>🧠 AI Internal Thoughts & Reasoning</h3>
                    <p>The cognitive process before each persona's response</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display thoughts/reasoning from transcript
                for i, msg in enumerate(results['transcript']):
                    speaker = msg.get('speaker', 'Unknown')
                    
                    # Check for various thought fields
                    thoughts = msg.get('thoughts', msg.get('thinking', msg.get('reasoning', msg.get('internal_monologue', None))))
                    
                    if thoughts:
                        with st.container():
                            st.markdown(f"""
                            <div class="glass-card" style="margin: 1rem 0; padding: 1rem; background: linear-gradient(135deg, rgba(118, 75, 162, 0.05), rgba(102, 126, 234, 0.05));">
                                <strong style="color: #764ba2;">🧠 {speaker}'s Thoughts</strong>
                                <p style="margin: 0.5rem 0 0 0; line-height: 1.6; font-style: italic; color: #555555 !important;">{thoughts}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        # If no explicit thoughts field, show a note
                        if i == 0:  # Only show once
                            st.info("💡 Thoughts data not captured in this simulation. Enable detailed logging to see internal reasoning.")
                            break
            
            with tab3:
                st.markdown("""
                <div class="glass-card">
                    <h3>💭 Chat with Individual Agents</h3>
                    <p>Ask follow-up questions to individual personas</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Get agents from session state
                agents = st.session_state.get('last_simulation_agents', [])
                
                if agents:
                    # Agent selector with group option - use display names
                    agent_names = ["🌐 All Agents (Group Chat)"] + [getattr(agent, '_display_name', agent.name) for agent in agents]
                    selected_agent_name = st.selectbox(
                        "Select an agent or group to chat with:",
                        agent_names,
                        key="agent_chat_selector"
                    )
                    
                    # Check if group chat is selected
                    is_group_chat = selected_agent_name == "🌐 All Agents (Group Chat)"
                    
                    if is_group_chat:
                        # Group chat mode
                        st.markdown("""
                        <div class="glass-card" style="margin: 1rem 0;">
                            <strong>🌐 Group Chat Mode</strong> - All agents will respond in turn
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Initialize chat history for group
                        chat_key = "agent_chat_group"
                        if chat_key not in st.session_state:
                            st.session_state[chat_key] = []
                        
                        # Display chat history
                        for chat_msg in st.session_state[chat_key]:
                            with st.container():
                                if chat_msg['role'] == 'user':
                                    st.markdown(f"""
                                    <div class="glass-card" style="margin: 0.5rem 0; padding: 0.8rem; background: rgba(100, 150, 255, 0.1);">
                                        <strong>You:</strong> {chat_msg['content']}
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.markdown(f"""
                                    <div class="glass-card" style="margin: 0.5rem 0; padding: 0.8rem;">
                                        <strong>{chat_msg['speaker']}:</strong> {chat_msg['content']}
                                    </div>
                                    """, unsafe_allow_html=True)
                        
                        # Question input for group
                        with st.form(key="agent_chat_form_group", clear_on_submit=True):
                            user_question = st.text_area(
                                "Ask a question to all agents:",
                                height=100,
                                key="agent_question_group"
                            )
                            submit_question = st.form_submit_button("💬 Send to All")
                            
                            if submit_question and user_question:
                                with st.spinner("💭 All agents are thinking..."):
                                    # Add user question
                                    st.session_state[chat_key].append({
                                        'role': 'user',
                                        'content': user_question
                                    })
                                    
                                    # Get responses from all agents
                                    for agent in agents:
                                        try:
                                            agent.listen_and_act(user_question)
                                            response = agent.act(return_actions=False)
                                            
                                            if response:
                                                display_name = getattr(agent, '_display_name', agent.name)
                                                st.session_state[chat_key].append({
                                                    'role': 'agent',
                                                    'speaker': display_name,
                                                    'content': response
                                                })
                                        except Exception as e:
                                            display_name = getattr(agent, '_display_name', agent.name)
                                            st.error(f"Error getting response from {display_name}: {str(e)}")
                                    
                                    st.rerun()
                        
                        # Clear chat button
                        if st.button("🗑️ Clear Group Chat History", key="clear_chat_group"):
                            st.session_state[chat_key] = []
                            st.rerun()
                    
                    else:
                        # Individual agent chat mode
                        # Find selected agent by display name
                        selected_agent = None
                        for agent in agents:
                            display_name = getattr(agent, '_display_name', agent.name)
                            if display_name == selected_agent_name:
                                selected_agent = agent
                                break
                        
                        if selected_agent:
                            # Display agent info
                            try:
                                role = getattr(selected_agent, 'role', 'Unknown')
                                if hasattr(selected_agent, '_persona') and isinstance(selected_agent._persona, dict):
                                    role = selected_agent._persona.get('role', role)
                            except:
                                role = 'Unknown'
                            
                            display_name = getattr(selected_agent, '_display_name', selected_agent.name)
                            
                            st.markdown(f"""
                            <div class="glass-card" style="margin: 1rem 0;">
                                <strong>👤 {display_name}</strong> - {role}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Initialize chat history for this agent (use original_id for key stability)
                            agent_id = getattr(selected_agent, '_original_id', selected_agent.name)
                            chat_key = f"agent_chat_{agent_id}"
                            if chat_key not in st.session_state:
                                st.session_state[chat_key] = []
                            
                            # Display chat history
                            for chat_msg in st.session_state[chat_key]:
                                with st.container():
                                    if chat_msg['role'] == 'user':
                                        st.markdown(f"""
                                        <div class="glass-card" style="margin: 0.5rem 0; padding: 0.8rem; background: rgba(100, 150, 255, 0.1);">
                                            <strong>You:</strong> {chat_msg['content']}
                                        </div>
                                        """, unsafe_allow_html=True)
                                    else:
                                        st.markdown(f"""
                                        <div class="glass-card" style="margin: 0.5rem 0; padding: 0.8rem;">
                                            <strong>{display_name}:</strong> {chat_msg['content']}
                                        </div>
                                        """, unsafe_allow_html=True)
                            
                            # Question input
                            with st.form(key=f"agent_chat_form_{agent_id}", clear_on_submit=True):
                                user_question = st.text_area(
                                    "Ask a follow-up question:",
                                    height=100,
                                    key=f"agent_question_{agent_id}"
                                )
                                submit_question = st.form_submit_button("💬 Send")
                                
                                if submit_question and user_question:
                                    with st.spinner(f"💭 {display_name} is thinking..."):
                                        try:
                                            # Send question to agent
                                            selected_agent.listen_and_act(user_question)
                                            response = selected_agent.act(return_actions=False)
                                            
                                            if response:
                                                # Add to chat history
                                                st.session_state[chat_key].append({
                                                    'role': 'user',
                                                    'content': user_question
                                                })
                                                st.session_state[chat_key].append({
                                                    'role': 'agent',
                                                    'content': response
                                                })
                                                st.rerun()
                                            else:
                                                st.warning("Agent did not provide a response.")
                                        except Exception as e:
                                            st.error(f"Error getting response: {str(e)}")
                            
                            # Clear chat button
                            if st.button("🗑️ Clear Chat History", key=f"clear_chat_{agent_id}"):
                                st.session_state[chat_key] = []
                                st.rerun()
                else:
                    st.info("💡 Run a simulation first to chat with agents.")
            
            with tab4:
                st.markdown("""
                <div class="glass-card">
                    <h3>📊 Conversation Analysis & Metrics</h3>
                    <p>Comprehensive insights from the AI panel discussion</p>
                </div>
                """, unsafe_allow_html=True)
                
                if 'analysis' in results and results['analysis']:
                    analysis = results['analysis']
                    
                    # Overview Metrics
                    if 'overview' in analysis:
                        st.subheader("📈 Overview Metrics")
                        overview = analysis['overview']
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Messages", overview.get('total_messages', 0))
                        with col2:
                            st.metric("Total Words", f"{overview.get('total_words', 0):,}")
                        with col3:
                            st.metric("Unique Speakers", overview.get('unique_speakers', 0))
                        with col4:
                            st.metric("Avg Message Length", f"{overview.get('avg_message_length', 0):.1f} words")
                    
                    # Participation Analysis
                    if 'participation' in analysis and 'by_speaker' in analysis['participation']:
                        st.subheader("👥 Participation Distribution")
                        
                        participation = analysis['participation']['by_speaker']
                        
                        # Create participation chart data
                        if participation:
                            import pandas as pd
                            
                            # Prepare data for visualization
                            speakers = []
                            messages = []
                            words = []
                            
                            for speaker, stats in participation.items():
                                speakers.append(speaker)
                                messages.append(stats.get('messages', 0))
                                words.append(stats.get('words', 0))
                            
                            # Display as table
                            df = pd.DataFrame({
                                'Persona': speakers,
                                'Messages': messages,
                                'Words': words,
                                'Avg Words/Message': [w/m if m > 0 else 0 for w, m in zip(words, messages)]
                            })
                            df = df.sort_values('Messages', ascending=False)
                            
                            st.dataframe(df, use_container_width=True, hide_index=True)
                            
                            # Show participation percentages
                            total_messages = sum(messages)
                            if total_messages > 0:
                                st.markdown("**Participation Percentages:**")
                                cols = st.columns(min(len(speakers), 4))
                                for idx, (speaker, msg_count) in enumerate(zip(speakers[:4], messages[:4])):
                                    with cols[idx]:
                                        percentage = (msg_count / total_messages) * 100
                                        st.metric(speaker, f"{percentage:.1f}%", delta=f"{msg_count} msgs")
                    
                    # Content Focus Analysis
                    if 'content_analysis' in analysis:
                        st.subheader("🎯 Content Focus Areas")
                        content = analysis['content_analysis']
                        
                        if 'focus_areas' in content:
                            focus_areas = content['focus_areas']
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("🔧 Technical", focus_areas.get('technical', 0), 
                                         help="Technical keywords and concepts discussed")
                            with col2:
                                st.metric("👤 UX/Design", focus_areas.get('ux', 0),
                                         help="User experience and design mentions")
                            with col3:
                                st.metric("💼 Business", focus_areas.get('business', 0),
                                         help="Business and strategy discussions")
                            
                            if 'dominant_focus' in content:
                                st.info(f"**Dominant Focus:** {content['dominant_focus'].title()}")
                    
                    # Interaction Patterns
                    if 'interaction_patterns' in analysis:
                        st.subheader("🔄 Interaction Patterns")
                        patterns = analysis['interaction_patterns']
                        
                        if 'most_active_pairs' in patterns and patterns['most_active_pairs']:
                            st.markdown("**Most Active Conversation Pairs:**")
                            
                            for pattern, count in patterns['most_active_pairs'][:5]:
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.markdown(f"• {pattern}")
                                with col2:
                                    st.markdown(f"**{count}** exchanges")
                    
                    # Turn Analysis
                    if 'turn_analysis' in analysis:
                        st.subheader("🔄 Turn-by-Turn Analysis")
                        turn_data = analysis['turn_analysis']
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Turns", turn_data.get('total_turns', 0))
                        with col2:
                            st.metric("Avg Messages/Turn", f"{turn_data.get('avg_messages_per_turn', 0):.1f}")
                        
                        if 'turn_lengths' in turn_data:
                            with st.expander("View Turn Details"):
                                for turn, length in turn_data['turn_lengths'].items():
                                    st.markdown(f"**Turn {turn}:** {length} messages")
                    
                    # Key Insights (if available)
                    if 'key_insights' in analysis and analysis['key_insights']:
                        st.subheader("🔍 Key Insights")
                        for insight in analysis['key_insights']:
                            st.markdown(f"• {insight}")
                    
                    # Sentiment Analysis (if available)
                    if 'sentiment_analysis' in analysis:
                        st.subheader("😊 Sentiment Analysis")
                        sentiment = analysis['sentiment_analysis']
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Positive", f"{sentiment.get('positive', 0):.1%}")
                        with col2:
                            st.metric("Neutral", f"{sentiment.get('neutral', 0):.1%}")
                        with col3:
                            st.metric("Negative", f"{sentiment.get('negative', 0):.1%}")
                
                else:
                    st.info("📊 Running detailed conversation analysis...")
                    
                    # Generate analysis from transcript
                    if 'transcript' in results and results['transcript']:
                        from collections import defaultdict
                        
                        transcript = results['transcript']
                        
                        # Calculate basic metrics
                        total_messages = len(transcript)
                        speakers = set()
                        total_words = 0
                        speaker_stats = defaultdict(lambda: {"messages": 0, "words": 0})
                        
                        for msg in transcript:
                            speaker = msg.get('speaker', 'Unknown')
                            speakers.add(speaker)
                            text = msg.get('content', msg.get('message', ''))
                            words = len(text.split())
                            total_words += words
                            
                            speaker_stats[speaker]['messages'] += 1
                            speaker_stats[speaker]['words'] += words
                        
                        # Display basic analysis
                        st.subheader("📈 Quick Analysis")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Messages", total_messages)
                        with col2:
                            st.metric("Total Words", f"{total_words:,}")
                        with col3:
                            st.metric("Unique Speakers", len(speakers))
                        with col4:
                            avg_length = total_words / total_messages if total_messages > 0 else 0
                            st.metric("Avg Message Length", f"{avg_length:.1f} words")
                        
                        # Participation breakdown
                        st.subheader("👥 Participation Breakdown")
                        
                        import pandas as pd
                        df_data = []
                        for speaker, stats in speaker_stats.items():
                            avg_words = stats['words'] / stats['messages'] if stats['messages'] > 0 else 0
                            df_data.append({
                                'Persona': speaker,
                                'Messages': stats['messages'],
                                'Words': stats['words'],
                                'Avg Words/Message': f"{avg_words:.1f}"
                            })
                        
                        df = pd.DataFrame(df_data)
                        df = df.sort_values('Messages', ascending=False)
                        st.dataframe(df, use_container_width=True, hide_index=True)
                    else:
                        st.warning("No transcript data available for analysis")
            
                with tab5:
                    st.markdown("""
                    <div class="glass-card">
                        <h3>📈 Performance Metrics</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    metrics = results.get('metrics', {})
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Duration", f"{metrics.get('duration', 0):.1f}s")
                    with col2:
                        st.metric("Messages", len(results['transcript']))
                    with col3:
                        st.metric("Participants", len(set(msg.get('speaker', '') for msg in results['transcript'])))
                    with col4:
                        st.metric("Avg Response Time", f"{metrics.get('avg_response_time', 0):.2f}s")
                    
                    # Token usage
                    if 'token_usage' in metrics:
                        token_data = metrics['token_usage']
                        st.subheader("🔢 Token Usage")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Prompt Tokens", f"{token_data.get('prompt_tokens', 0):,}")
                        with col2:
                            st.metric("Completion Tokens", f"{token_data.get('completion_tokens', 0):,}")
                        with col3:
                            st.metric("Total Tokens", f"{token_data.get('total_tokens', 0):,}")
            
            with tab6:
                st.markdown("""
                <div class="glass-card">
                    <h3>📁 Export Options</h3>
                    <p>Download your simulation results in various formats</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Export buttons
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📄 Export as JSON", use_container_width=True):
                        json_str = json.dumps(results, indent=2, ensure_ascii=False)
                        st.download_button(
                            "Download JSON",
                            json_str,
                            f"simulation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            "application/json"
                        )
                
                with col2:
                    if st.button("📝 Export as Markdown", use_container_width=True):
                        md_content = f"# Persona Panel Discussion\\n\\n## Topic\\n{seed}\\n\\n## Conversation\\n\\n"
                        for msg in results['transcript']:
                            speaker = msg.get('speaker', 'Unknown')
                            content = msg.get('content', msg.get('message', ''))
                            md_content += f"**{speaker}:** {content}\\n\\n"
                        
                        st.download_button(
                            "Download Markdown",
                            md_content,
                            f"discussion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                            "text/markdown"
                        )
                
                with col3:
                    st.info("💡 More export formats coming soon!")
        
        else:
            st.markdown("""
            <div class="glass-card" style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);">
                <h3>❌ Simulation Failed</h3>
                <p>The simulation encountered an error. Please check your configuration and try again.</p>
            </div>
            """, unsafe_allow_html=True)

# ---------- Footer ----------
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #333333;">
    <p>Built with ❤️ using <strong>TinyTroupe</strong> & <strong>Streamlit</strong></p>
    <p>Enhanced with AI-powered features for intelligent persona panel simulations</p>
</div>
""", unsafe_allow_html=True)