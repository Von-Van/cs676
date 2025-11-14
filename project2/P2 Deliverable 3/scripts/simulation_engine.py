#!/usr/bin/env python3
"""
simulation_engine.py — Consolidated TinyTroupe Simulation Engine

A production-ready, scalable simulation engine that consolidates all previous
script functionalities into a single, comprehensive tool.

Features:
- Comprehensive persona database management with validation
- Multi-mode simulation (single run, batch experiments, interactive)
- Real-time conversation analysis and visualization
- Load balancing and error handling
- Monitoring and performance metrics
- Automatic report generation
- Configurable scaling parameters

Usage:
    # Single simulation
    python simulation_engine.py --mode single --seed "Your discussion topic"
    
    # Batch experiments
    python simulation_engine.py --mode experiment --variants baseline improved --runs 3
    
    # Interactive mode (for Streamlit integration)
    from simulation_engine import SimulationEngine
    engine = SimulationEngine()
    results = engine.run_simulation(seed="topic", turns=6)
"""

import os
import json
import time
import random
import copy
import logging
import argparse
import configparser
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
import queue

# Core TinyTroupe imports
from tinytroupe.agent import TinyPerson
from tinytroupe.environment import TinyWorld
from tinytroupe import config_manager
from tinytroupe.openai_utils import force_api_cache
from tinytroupe.agent.memory import EpisodicMemory, SemanticMemory
from tinytroupe.agent.mental_faculty import RecallFaculty, TinyToolUse
from tinytroupe.factory import TinyPersonFactory
from tinytroupe.tools import TinyWordProcessor

# Persona validation
try:
    from scripts.persona_validator import PersonaValidator, ValidationResult
    HAS_VALIDATOR = True
except ImportError:
    HAS_VALIDATOR = False

# Optional analysis dependencies
try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False

# ==================== CONFIGURATION ====================

@dataclass
class SimulationConfig:
    """Centralized configuration for simulation parameters."""
    # Core simulation settings
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_turns: int = 6
    seed: str = ""
    
    # Persona settings
    personas_file: str = "data/personas.agents.json"
    max_personas: int = 10
    persona_timeout: float = 30.0
    
    # Performance & scaling
    parallel_execution: bool = False
    max_workers: int = 3
    rate_limit_delay: float = 1.0
    max_retries: int = 5
    batch_size: int = 5
    
    # Monitoring & logging
    enable_monitoring: bool = True
    log_level: str = "INFO"
    performance_tracking: bool = True
    
    # Output settings
    output_dir: str = "outputs"
    auto_analysis: bool = True
    save_visualizations: bool = True
    
    # Error handling
    timeout_seconds: int = 300
    graceful_degradation: bool = True
    fallback_model: str = "gpt-3.5-turbo"
    
    # Memory and cognitive features
    enable_session_memory: bool = True
    enable_recall_faculty: bool = True
    enable_collaborative_tools: bool = True
    memory_retention_turns: int = 50
    
    # Agent selection
    enable_smart_agent_selection: bool = True
    max_selected_agents: int = 5
    agent_selection_model: str = "gpt-4o-mini"
    
    # Token limits
    max_tokens_per_turn: int = 12000
    max_completion_tokens: int = 8000

@dataclass
class PersonaSpec:
    """Enhanced persona specification with validation."""
    id: str
    name: str
    role: str
    description: str
    personality: Dict[str, Any]
    model: str = ""
    temperature: float = 0.7
    prompting: Dict[str, Any] = field(default_factory=dict)
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> List[str]:
        """Validate persona specification and return any issues."""
        issues = []
        
        # Required field validation
        required_fields = ['id', 'name', 'role', 'description']
        for field_name in required_fields:
            if not getattr(self, field_name, "").strip():
                issues.append(f"Missing or empty required field: {field_name}")
        
        # Personality structure validation
        if not isinstance(self.personality, dict):
            issues.append("Personality must be a dictionary")
        else:
            required_personality = ['traits', 'goals', 'style']
            for field_name in required_personality:
                if field_name not in self.personality:
                    issues.append(f"Missing personality field: {field_name}")
        
        # Temperature range validation
        if not (0.0 <= self.temperature <= 2.0):
            issues.append(f"Temperature {self.temperature} out of range [0.0, 2.0]")
        
        # ID format validation
        if not self.id.replace('_', '').replace('-', '').isalnum():
            issues.append(f"Invalid ID format: {self.id}. Use alphanumeric, underscore, dash only")
        
        return issues

@dataclass
class SimulationMetrics:
    """Performance and monitoring metrics."""
    start_time: float = 0.0
    end_time: float = 0.0
    total_messages: int = 0
    total_tokens: int = 0
    api_calls: int = 0
    errors: int = 0
    timeouts: int = 0
    average_response_time: float = 0.0
    personas_active: int = 0
    turns_completed: int = 0
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time if self.end_time > self.start_time else 0.0
    
    @property
    def messages_per_minute(self) -> float:
        return (self.total_messages / (self.duration / 60)) if self.duration > 0 else 0.0

# ==================== PERSONA DATABASE MANAGER ====================

class PersonaDatabase:
    """Enhanced persona database with validation, caching, and expansion capabilities."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.personas_cache = {}
        self.validation_enabled = True
        self.logger = logging.getLogger(__name__ + ".PersonaDatabase")
        
    def load_personas(self, personas_file: Optional[Path] = None) -> List[PersonaSpec]:
        """Load and validate personas from JSON file with enhanced error handling."""
        if personas_file is None:
            personas_file = Path(self.config.personas_file)
        
        if not personas_file.exists():
            raise FileNotFoundError(f"Personas file not found: {personas_file}")
        
        try:
            with personas_file.open('r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in personas file: {e}")
        
        if not isinstance(data, list):
            raise ValueError("Personas file must contain a list of persona objects")
        
        personas = []
        validation_errors = []
        
        for i, spec_dict in enumerate(data):
            try:
                # Convert dict to PersonaSpec with defaults
                persona = PersonaSpec(
                    id=spec_dict.get('id', f'persona_{i}'),
                    name=spec_dict.get('name', f'Agent {i+1}'),
                    role=spec_dict.get('role', 'Participant'),
                    description=spec_dict.get('description', ''),
                    personality=spec_dict.get('personality', {}),
                    model=spec_dict.get('model', self.config.model),
                    temperature=spec_dict.get('temperature', self.config.temperature),
                    prompting=spec_dict.get('prompting', {}),
                    validation_rules=spec_dict.get('validation_rules', {})
                )
                
                # Enhanced validation with consistency and realism checking
                if self.validation_enabled and HAS_VALIDATOR:
                    try:
                        validator = PersonaValidator()
                        validation_result = validator.validate_persona_structure(spec_dict)
                        
                        if not validation_result.is_valid:
                            for issue in validation_result.issues:
                                validation_errors.append(f"Persona {i} ({persona.name}): {issue}")
                        
                        # Log warnings and suggestions
                        if validation_result.warnings:
                            for warning in validation_result.warnings:
                                self.logger.warning(f"Persona {persona.name}: {warning}")
                        
                        if validation_result.consistency_score < 0.7:
                            self.logger.warning(
                                f"Persona {persona.name} has low consistency score: "
                                f"{validation_result.consistency_score:.2f}"
                            )
                        
                        if validation_result.realism_score < 0.7:
                            self.logger.warning(
                                f"Persona {persona.name} has low realism score: "
                                f"{validation_result.realism_score:.2f}"
                            )
                    except Exception as val_error:
                        self.logger.warning(f"Validation failed for {persona.name}: {val_error}")
                
                personas.append(persona)
                self.personas_cache[persona.id] = persona
                self.logger.debug(f"Successfully loaded persona: {persona.name} (ID: {persona.id})")
                
            except Exception as e:
                validation_errors.append(f"Persona {i}: Error creating persona - {str(e)}")
                self.logger.error(f"Failed to create persona {i}: {e}")
        
        if validation_errors:
            error_msg = "Persona validation errors:\n" + "\n".join(validation_errors)
            if not self.config.graceful_degradation:
                raise ValueError(error_msg)
            else:
                self.logger.warning(error_msg)
        
        if not personas:
            raise ValueError("No valid personas found after validation")
        
        self.logger.info(f"Loaded {len(personas)} personas from {personas_file}")
        return personas
    
    def get_all_personas(self) -> Dict[str, PersonaSpec]:
        """Get all loaded personas as a dictionary mapping names to PersonaSpec objects."""
        self.logger.debug(f"get_all_personas called, cache size: {len(self.personas_cache)}")
        
        if not self.personas_cache:
            # Load personas if cache is empty
            try:
                self.logger.debug("Cache is empty, loading personas...")
                personas_list = self.load_personas()
                self.logger.debug(f"load_personas returned {len(personas_list)} personas")
                # Cache should already be populated by load_personas, but double-check
                for persona in personas_list:
                    self.personas_cache[persona.id] = persona
                    self.logger.debug(f"Added to cache: {persona.name} (ID: {persona.id})")
            except Exception as e:
                self.logger.error(f"Failed to load personas: {e}")
                return {}
        
        self.logger.debug(f"Returning {len(self.personas_cache)} personas from cache")
        result = {persona.name: persona for persona in self.personas_cache.values()}
        self.logger.debug(f"Result dict has {len(result)} entries: {list(result.keys())}")
        return result
    
    def get_persona_by_id(self, persona_id: str) -> Optional[PersonaSpec]:
        """Get a specific persona by ID."""
        return self.personas_cache.get(persona_id)
    
    def create_tiny_persons(self, persona_specs: List[PersonaSpec]) -> List[TinyPerson]:
        """Convert PersonaSpecs to TinyPerson objects with robust error handling."""
        # Clear the global TinyPerson registry to allow agent reuse
        try:
            if hasattr(TinyPerson, 'all_agents'):
                TinyPerson.all_agents.clear()
            if hasattr(TinyPerson, '_name_to_agent'):
                TinyPerson._name_to_agent.clear()
            self.logger.debug("Cleared TinyPerson global registry")
        except Exception as e:
            self.logger.warning(f"Could not clear TinyPerson registry: {e}")
        
        agents = []
        timestamp_suffix = datetime.now().strftime('%H%M%S%f')[:10]
        
        for idx, spec in enumerate(persona_specs):
            try:
                # Create unique agent name to prevent collisions
                unique_name = f"{spec.id}_{timestamp_suffix}_{idx}"
                
                # Create TinyPerson with proper initialization
                agent = TinyPerson(name=unique_name)
                
                # Store original name for display purposes
                agent._display_name = spec.name
                agent._original_id = spec.id
                
                # Configure core attributes
                agent.define("role", spec.role)
                agent.define("description", spec.description)
                
                # Set personality details
                if isinstance(spec.personality, dict):
                    agent.define("personality", spec.personality)
                
                # Configure model parameters
                agent.define("model", spec.model or self.config.model)
                agent.define("temperature", spec.temperature)
                
                # Set prompting configuration
                if spec.prompting:
                    agent.define("prompting", spec.prompting)
                
                # Add additional biographical info expected by TinyPerson
                agent.define("occupation", spec.role)
                agent.define("residence", "Unknown")
                agent.define("age", None)
                
                agents.append(agent)
                self.logger.debug(f"Created agent: {spec.name} (internal: {unique_name})")
                
            except Exception as e:
                error_msg = f"Failed to create TinyPerson for {spec.name}: {str(e)}"
                if self.config.graceful_degradation:
                    self.logger.error(error_msg)
                    continue
                else:
                    raise ValueError(error_msg)
        
        if not agents:
            raise ValueError("No agents could be created from persona specifications")
        
        return agents
    
    def save_personas(self, personas: List[PersonaSpec], output_file: Path):
        """Save personas to JSON file with backup."""
        # Create backup if file exists
        if output_file.exists():
            backup_path = output_file.with_suffix(f'.backup.{int(time.time())}.json')
            output_file.rename(backup_path)
            self.logger.info(f"Created backup: {backup_path}")
        
        # Convert PersonaSpec objects to dicts
        data = []
        for persona in personas:
            persona_dict = {
                'type': 'TinyPerson',
                'id': persona.id,
                'name': persona.name,
                'role': persona.role,
                'description': persona.description,
                'personality': persona.personality,
                'model': persona.model,
                'temperature': persona.temperature,
                'prompting': persona.prompting
            }
            if persona.validation_rules:
                persona_dict['validation_rules'] = persona.validation_rules
            data.append(persona_dict)
        
        with output_file.open('w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved {len(personas)} personas to {output_file}")
    
    def setup_session_memory(self, agent, session_id: str):
        """Setup session memory for an agent (placeholder implementation)."""
        try:
            # For now, just return the agent as-is since TinyTroupe handles memory internally
            self.logger.debug(f"Setting up session memory for {getattr(agent, 'name', 'unknown')} in session {session_id}")
            return agent
        except Exception as e:
            self.logger.warning(f"Failed to setup session memory: {e}")
            return agent
    
    def clear_session_memory(self, agents):
        """Clear session memory for agents and reset them for reuse."""
        try:
            self.logger.debug(f"Clearing session memory for {len(agents)} agents")
            for agent in agents:
                # Clear agent's episodic memory
                if hasattr(agent, '_episodic_memory'):
                    agent._episodic_memory = []
                # Clear agent's semantic memory
                if hasattr(agent, '_semantic_memory'):
                    agent._semantic_memory = {}
                # Clear any cached messages
                if hasattr(agent, '_messages'):
                    agent._messages = []
                # Reset current context
                if hasattr(agent, '_current_context'):
                    agent._current_context = None
                self.logger.debug(f"Cleared memory for agent: {agent.name}")
        except Exception as e:
            self.logger.warning(f"Failed to clear session memory: {e}")
    
    def reset_agents_for_reuse(self, agents, world=None):
        """Completely reset agents for reuse in a new simulation."""
        try:
            self.logger.debug(f"Resetting {len(agents)} agents for reuse")
            
            # Remove agents from world if provided
            if world and hasattr(world, 'agents'):
                world.agents = []
            
            # Clear all agent memories and state
            for agent in agents:
                # Clear episodic memory
                if hasattr(agent, '_episodic_memory'):
                    agent._episodic_memory = []
                # Clear semantic memory  
                if hasattr(agent, '_semantic_memory'):
                    agent._semantic_memory = {}
                # Clear cached messages
                if hasattr(agent, '_messages'):
                    agent._messages = []
                # Clear current context
                if hasattr(agent, '_current_context'):
                    agent._current_context = None
                # Clear any actions
                if hasattr(agent, '_actions'):
                    agent._actions = []
                # Reset conversation history
                if hasattr(agent, 'conversation_history'):
                    agent.conversation_history = []
                # Reset any world reference
                if hasattr(agent, '_world'):
                    agent._world = None
                    
                self.logger.debug(f"Reset agent {agent.name} for reuse")
                
            return True
        except Exception as e:
            self.logger.error(f"Failed to reset agents for reuse: {e}")
            return False

# ==================== SIMULATION ENGINE ====================

class AgentSelector:
    """Intelligent agent selection based on topic relevance."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def select_agents_for_topic(self, topic: str, available_personas: List[PersonaSpec], max_agents: Optional[int] = None) -> List[PersonaSpec]:
        """Select the most relevant agents for a given topic using AI."""
        max_agents = max_agents or self.config.max_selected_agents
        
        if len(available_personas) <= max_agents:
            return available_personas
        
        try:
            # Create agent descriptions for evaluation
            agent_descriptions = []
            for persona in available_personas:
                description = f"""
                Name: {persona.name}
                Role: {persona.role}
                Description: {persona.description}
                Personality: {persona.personality.get('traits', [])} - {persona.personality.get('style', '')}
                Goals: {persona.personality.get('goals', [])}
                """
                agent_descriptions.append({
                    "persona": persona,
                    "description": description.strip()
                })
            
            # Use LLM to select most relevant agents
            selected_personas = self._ai_agent_selection(topic, agent_descriptions, max_agents)
            
            self.logger.info(f"Selected {len(selected_personas)} agents for topic: {topic}")
            return selected_personas
            
        except Exception as e:
            self.logger.error(f"Agent selection failed: {e}. Using first {max_agents} agents.")
            return available_personas[:max_agents]
    
    def _ai_agent_selection(self, topic: str, agent_descriptions: List[Dict], max_agents: int) -> List[PersonaSpec]:
        """Use AI to intelligently select agents based on topic relevance."""
        import tinytroupe.openai_utils as openai_utils
        
        # Prepare agent list for the prompt
        agent_list = ""
        for i, desc in enumerate(agent_descriptions):
            agent_list += f"{i+1}. {desc['description']}\n\n"
        
        prompt = f"""
        You are an expert team composition advisor. Given a discussion topic and a list of available team members, select the {max_agents} most relevant and diverse participants who would contribute meaningfully to the discussion.
        
        DISCUSSION TOPIC: {topic}
        
        AVAILABLE TEAM MEMBERS:
        {agent_list}
        
        SELECTION CRITERIA:
        1. Expertise relevance to the topic
        2. Diverse perspectives and backgrounds
        3. Complementary skills and viewpoints
        4. Potential for meaningful contribution
        5. Balanced team composition
        
        Please select exactly {max_agents} team members by returning ONLY their numbers (e.g., "1,3,5,7,9").
        No explanations or additional text - just the comma-separated numbers.
        """
        
        try:
            messages = [
                {"role": "system", "content": "You are a team composition expert. Respond only with comma-separated numbers."},
                {"role": "user", "content": prompt}
            ]
            
            response = openai_utils.client().send_message(
                messages, 
                model=self.config.agent_selection_model,
                temperature=0.3,
                max_tokens=2000  # Increased for better agent selection
            )
            
            if response and response.get("content"):
                # Parse the response
                selected_indices = []
                for num_str in response["content"].strip().split(","):
                    try:
                        idx = int(num_str.strip()) - 1  # Convert to 0-based index
                        if 0 <= idx < len(agent_descriptions):
                            selected_indices.append(idx)
                    except ValueError:
                        continue
                
                # Ensure we don't exceed max_agents
                selected_indices = selected_indices[:max_agents]
                
                # If we don't have enough, fill with first available
                while len(selected_indices) < min(max_agents, len(agent_descriptions)):
                    for i in range(len(agent_descriptions)):
                        if i not in selected_indices:
                            selected_indices.append(i)
                            break
                    break
                
                return [agent_descriptions[i]["persona"] for i in selected_indices]
        
        except Exception as e:
            self.logger.warning(f"AI agent selection failed: {e}")
        
        # Fallback: return first max_agents
        return [desc["persona"] for desc in agent_descriptions[:max_agents]]

class SimulationEngine:
    """Main simulation engine with comprehensive features."""
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        self.persona_db = PersonaDatabase(self.config)
        self.agent_selector = AgentSelector(self.config) if self.config.enable_smart_agent_selection else None
        self.metrics = SimulationMetrics()
        self.logger = self._setup_logging()
        self.session_id = None
        
        # Performance tracking
        self.response_times = []
        self.error_log = []
        
        # Rate limiting
        self.last_api_call = 0.0
        self.api_call_count = 0
        
        # Load balancing
        self.worker_pool = None
        
        # Initialize personas cache
        try:
            self.persona_db.get_all_personas()  # This will trigger loading
            self.logger.info(f"Initialized with {len(self.persona_db.personas_cache)} personas")
        except Exception as e:
            self.logger.warning(f"Failed to preload personas: {e}")
        if self.config.parallel_execution:
            self.worker_pool = ThreadPoolExecutor(max_workers=self.config.max_workers)
    
    def _setup_logging(self) -> logging.Logger:
        """Configure logging with appropriate level and formatting."""
        logger = logging.getLogger(__name__ + ".SimulationEngine")
        logger.setLevel(getattr(logging, self.config.log_level.upper(), logging.INFO))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _apply_rate_limiting(self):
        """Apply rate limiting to prevent API overuse."""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call
        
        if time_since_last_call < self.config.rate_limit_delay:
            sleep_time = self.config.rate_limit_delay - time_since_last_call
            time.sleep(sleep_time)
        
        self.last_api_call = time.time()
        self.api_call_count += 1
    
    def _handle_agent_action(self, agent: TinyPerson, prompt: str) -> Optional[str]:
        """Execute agent action with error handling and retries."""
        for attempt in range(self.config.max_retries):
            try:
                self._apply_rate_limiting()
                start_time = time.time()
                
                # Temporarily disable memory consolidation to avoid library issues
                original_consolidate = getattr(agent, "consolidate_episode_memories", None)
                if callable(original_consolidate):
                    agent.consolidate_episode_memories = lambda: None
                
                try:
                    result = agent.listen_and_act(prompt, return_actions=True)
                    response_time = time.time() - start_time
                    self.response_times.append(response_time)
                    
                    # Convert result to text
                    text = self._safe_to_text(result)
                    return text.strip() if text else None
                    
                finally:
                    # Restore original function
                    if callable(original_consolidate):
                        agent.consolidate_episode_memories = original_consolidate
                
            except Exception as e:
                self.metrics.errors += 1
                error_details = {
                    'agent': agent.name,
                    'attempt': attempt + 1,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                self.error_log.append(error_details)
                
                # Check if it's a rate limit error
                if "429" in str(e).lower() or "rate" in str(e).lower():
                    wait_time = min(5 * (2 ** attempt), 60)
                    self.logger.warning(f"Rate limit hit for {agent.name}, waiting {wait_time}s")
                    time.sleep(wait_time)
                    continue
                
                # Check if it's a timeout
                if "timeout" in str(e).lower():
                    self.metrics.timeouts += 1
                
                if attempt == self.config.max_retries - 1:
                    if self.config.graceful_degradation:
                        self.logger.error(f"Agent {agent.name} failed after {self.config.max_retries} attempts: {e}")
                        return None
                    else:
                        raise
                
                # Exponential backoff for other errors
                wait_time = 2 ** attempt
                time.sleep(wait_time)
        
        return None
    
    def _safe_to_text(self, obj) -> str:
        """Convert TinyTroupe return types to plain text safely."""
        if obj is None:
            return ""
        if isinstance(obj, str):
            return obj
        
        # Handle list of objects
        if isinstance(obj, list):
            parts = [self._safe_to_text(item) for item in obj]
            return "\n\n".join([p for p in parts if p])
        
        # Handle dictionary objects
        if isinstance(obj, dict):
            # Try common content keys
            for key in ("content", "text", "message", "body"):
                if key in obj:
                    return self._safe_to_text(obj[key])
            
            # Handle nested action structure
            if "action" in obj and isinstance(obj["action"], dict):
                return self._safe_to_text(obj["action"].get("content", ""))
        
        # Try text attribute
        try:
            text = getattr(obj, "text", None)
            if isinstance(text, str):
                return text
        except Exception:
            pass
        
        # Fallback to string representation
        try:
            return str(obj)
        except Exception:
            return ""
    
    def run_simulation(self, 
                      seed: str, 
                      turns: Optional[int] = None,
                      personas: Optional[List[str]] = None,
                      custom_personas: Optional[List[PersonaSpec]] = None,
                      use_smart_selection: Optional[bool] = None,
                      environment_name: Optional[str] = None,
                      environment_description: Optional[str] = None,
                      custom_context: Optional[str] = None,
                      discussion_goal: Optional[str] = None) -> Dict[str, Any]:
        """Run a single simulation with comprehensive monitoring."""
        
        self.metrics = SimulationMetrics()
        self.metrics.start_time = time.time()
        
        # Generate session ID for memory management
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
        
        try:
            # Load personas
            if custom_personas:
                available_personas = custom_personas
            else:
                available_personas = self.persona_db.load_personas()
            
            # Intelligent agent selection if enabled
            if use_smart_selection is None:
                use_smart_selection = self.config.enable_smart_agent_selection
            
            if use_smart_selection and self.agent_selector and not personas:
                self.logger.info(f"Using intelligent agent selection for topic: {seed[:100]}...")
                selected_personas = self.agent_selector.select_agents_for_topic(
                    seed, 
                    available_personas,
                    self.config.max_personas
                )
                persona_specs = selected_personas
                
                # Log selected agents
                selected_names = [p.name for p in selected_personas]
                self.logger.info(f"Selected agents: {', '.join(selected_names)}")
                
            elif personas:
                # Filter by specified names
                persona_specs = [p for p in available_personas if p.name in personas or p.id in personas]
                if len(persona_specs) != len(personas):
                    found_names = {p.name for p in persona_specs}
                    missing = set(personas) - found_names
                    self.logger.warning(f"Some personas not found: {missing}")
            else:
                # Use all available personas (limited by config)
                persona_specs = available_personas[:self.config.max_personas]
            
            if not persona_specs:
                raise ValueError("No personas available for simulation")
            
            # Create TinyPerson agents with session memory
            agents = self.persona_db.create_tiny_persons(persona_specs)
            
            # Setup session memory for each agent
            if self.config.enable_session_memory:
                try:
                    agents = [self.persona_db.setup_session_memory(agent, self.session_id) for agent in agents]
                except Exception as memory_error:
                    self.logger.warning(f"Session memory setup failed: {memory_error}")
            
            self.metrics.personas_active = len(agents)
            
            # Use custom environment settings or defaults with unique suffix to avoid reuse errors
            base_env_name = environment_name or "DiscussionRoom"
            env_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            env_name = f"{base_env_name}_{env_timestamp}"
            env_description = environment_description or "A professional meeting room designed for collaborative discussions."
            goal = discussion_goal or "Participate constructively in the discussion. Be concrete and practical."
            
            # Clear TinyWorld global registry if it exists
            try:
                if hasattr(TinyWorld, 'all_environments'):
                    TinyWorld.all_environments.clear()
                if hasattr(TinyWorld, '_name_to_world'):
                    TinyWorld._name_to_world.clear()
                self.logger.debug("Cleared TinyWorld global registry")
            except Exception as e:
                self.logger.warning(f"Could not clear TinyWorld registry: {e}")
            
            # Set up world environment
            world = TinyWorld(
                name=env_name,
                agents=agents,
                initial_datetime=datetime.now()
            )
            
            self.logger.debug(f"Created world: {base_env_name} (internal: {env_name})")
            
            try:
                world.make_everyone_accessible()
            except Exception as e:
                self.logger.warning(f"Could not make agents accessible: {e}")
            
            # Set context
            context_lines = [
                f"Environment: {env_description}",
                f"Topic: {seed}"
            ]
            
            # Add custom context if provided
            if custom_context and custom_context.strip():
                context_lines.append(f"Additional Context: {custom_context.strip()}")
            
            try:
                world.broadcast_context_change(context_lines)
            except Exception as e:
                self.logger.warning(f"Could not set context: {e}")
                # Fallback to regular broadcast
                for line in context_lines:
                    world.broadcast(line)
            
            # Set initial goal
            try:
                world.broadcast_internal_goal(goal)
            except Exception as e:
                self.logger.warning(f"Could not set internal goal: {e}")
            
            # Move agents to discussion environment
            for agent in agents:
                try:
                    agent.move_to(env_name)
                except Exception as e:
                    self.logger.warning(f"Could not move {agent.name}: {e}")
            
            # Start discussion
            world.broadcast(seed)
            
            # Run simulation turns
            transcript = []
            turns = turns or self.config.max_turns
            
            for turn in range(1, turns + 1):
                self.logger.info(f"Starting turn {turn}/{turns}")
                
                for agent in agents:
                    prompt = "Contribute one concise, concrete point and hand off implicitly."
                    
                    # Execute agent action
                    text = self._handle_agent_action(agent, prompt)
                    
                    if text:
                        # Get agent role
                        role = "Unknown"
                        try:
                            if hasattr(agent, "_persona") and isinstance(agent._persona, dict):
                                role = agent._persona.get("role", "Unknown")
                            else:
                                role = getattr(agent, "role", "Unknown")
                        except Exception:
                            pass
                        
                        # Capture AI thoughts from episodic memory
                        thoughts = None
                        try:
                            if hasattr(agent, '_episodic_memory') and agent._episodic_memory:
                                # Get the most recent memory entry
                                recent_memory = agent._episodic_memory[-1]
                                if isinstance(recent_memory, dict):
                                    thoughts = recent_memory.get('stimulus', '')
                                elif isinstance(recent_memory, str):
                                    thoughts = recent_memory
                        except Exception as thought_error:
                            self.logger.debug(f"Could not extract thoughts: {thought_error}")
                        
                        # Use display name if available, otherwise use agent name
                        speaker_name = getattr(agent, '_display_name', agent.name)
                        
                        transcript.append({
                            "turn": turn,
                            "speaker": speaker_name,
                            "role": role,
                            "content": text,  # Changed from "text" to "content" to match UI expectation
                            "text": text,     # Keep both for compatibility
                            "thoughts": thoughts,  # Add captured thoughts
                            "timestamp": datetime.now().isoformat()
                        })
                        
                        self.metrics.total_messages += 1
                        # Rough token estimation
                        self.metrics.total_tokens += len(text.split()) * 1.3
                
                self.metrics.turns_completed = turn
                
                # Add pacing between turns
                if turn < turns:
                    time.sleep(self.config.rate_limit_delay)
            
            self.metrics.end_time = time.time()
            
            # Calculate average response time
            if self.response_times:
                self.metrics.average_response_time = sum(self.response_times) / len(self.response_times)
            
            # Generate conversation analysis
            analysis = None
            if self.config.auto_analysis and transcript:
                try:
                    analysis = self.analyze_conversation(transcript)
                except Exception as analysis_error:
                    self.logger.warning(f"Analysis failed: {analysis_error}")
            
            # Prepare results
            results = {
                "success": True,
                "transcript": transcript,
                "agents": [{"name": getattr(a, '_display_name', a.name), "role": getattr(a, "role", "Unknown")} for a in agents],
                "agent_objects": agents,  # Include actual agent objects for reuse and chat
                "analysis": analysis,  # Add analysis to results
                "metrics": {
                    "duration": self.metrics.duration,
                    "total_messages": self.metrics.total_messages,
                    "total_tokens": int(self.metrics.total_tokens),
                    "api_calls": self.api_call_count,
                    "errors": self.metrics.errors,
                    "timeouts": self.metrics.timeouts,
                    "average_response_time": self.metrics.average_response_time,
                    "messages_per_minute": self.metrics.messages_per_minute,
                    "personas_active": self.metrics.personas_active,
                    "turns_completed": self.metrics.turns_completed
                },
                "config": {
                    "model": self.config.model,
                    "temperature": self.config.temperature,
                    "turns": turns,
                    "seed": seed,
                    "environment_name": env_name,
                    "environment_description": env_description,
                    "discussion_goal": goal,
                    "custom_context": custom_context
                },
                "session_id": self.session_id,
                "selected_agents": [getattr(agent, '_display_name', agent.name) for agent in agents] if use_smart_selection else None
            }
            
            self.logger.info(f"Simulation completed successfully in {self.metrics.duration:.1f}s")
            
            # Clean up agents and world for reuse
            try:
                self.persona_db.reset_agents_for_reuse(agents, world)
                self.logger.debug("Agents and world cleaned up for reuse")
            except Exception as cleanup_error:
                self.logger.warning(f"Failed to cleanup agents/world: {cleanup_error}")
            
            return results
            
        except Exception as e:
            self.metrics.end_time = time.time()
            error_result = {
                "success": False,
                "error": str(e),
                "metrics": {
                    "duration": self.metrics.duration,
                    "errors": self.metrics.errors + 1,
                    "personas_active": self.metrics.personas_active,
                    "turns_completed": self.metrics.turns_completed
                }
            }
            
            self.logger.error(f"Simulation failed: {e}")
            
            if not self.config.graceful_degradation:
                raise
            
            return error_result
        
        finally:
            # Clean up session memory if configured
            if hasattr(self, 'session_id') and self.config.enable_session_memory:
                try:
                    if 'agents' in locals():
                        self.persona_db.clear_session_memory(agents)
                except Exception as cleanup_error:
                    self.logger.warning(f"Failed to cleanup session memory: {cleanup_error}")
            
            self.logger.debug(f"Simulation cleanup completed for session {getattr(self, 'session_id', 'unknown')}")
    
    def run_experiments(self, 
                       variants: Dict[str, Dict[str, Any]], 
                       runs_per_variant: int = 1) -> Dict[str, Any]:
        """Run multiple experiment variants with comparison analysis."""
        
        experiment_start = time.time()
        all_results = []
        variant_results = {}
        
        self.logger.info(f"Starting experiment with {len(variants)} variants, {runs_per_variant} runs each")
        
        for variant_name, variant_config in variants.items():
            self.logger.info(f"Running variant: {variant_name}")
            
            variant_runs = []
            
            for run in range(runs_per_variant):
                self.logger.info(f"  Run {run + 1}/{runs_per_variant}")
                
                # Update config for this variant
                temp_config = copy.deepcopy(self.config)
                for key, value in variant_config.items():
                    if hasattr(temp_config, key):
                        setattr(temp_config, key, value)
                
                # Create new engine instance with variant config
                engine = SimulationEngine(temp_config)
                
                # Run simulation
                result = engine.run_simulation(
                    seed=variant_config.get("seed", self.config.seed),
                    turns=variant_config.get("max_turns", self.config.max_turns)
                )
                
                # Add variant information
                result["variant"] = variant_name
                result["run"] = run + 1
                
                variant_runs.append(result)
                all_results.append(result)
            
            variant_results[variant_name] = variant_runs
        
        experiment_duration = time.time() - experiment_start
        
        # Generate experiment summary
        summary = {
            "experiment_duration": experiment_duration,
            "total_runs": len(all_results),
            "variants": list(variants.keys()),
            "runs_per_variant": runs_per_variant,
            "results": all_results,
            "variant_summary": {}
        }
        
        # Calculate per-variant statistics
        for variant_name, runs in variant_results.items():
            successful_runs = [r for r in runs if r.get("success", False)]
            
            if successful_runs:
                avg_messages = sum(r["metrics"]["total_messages"] for r in successful_runs) / len(successful_runs)
                avg_duration = sum(r["metrics"]["duration"] for r in successful_runs) / len(successful_runs)
                
                summary["variant_summary"][variant_name] = {
                    "successful_runs": len(successful_runs),
                    "failed_runs": len(runs) - len(successful_runs),
                    "avg_messages": avg_messages,
                    "avg_duration": avg_duration,
                    "success_rate": len(successful_runs) / len(runs)
                }
            else:
                summary["variant_summary"][variant_name] = {
                    "successful_runs": 0,
                    "failed_runs": len(runs),
                    "success_rate": 0.0
                }
        
        self.logger.info(f"Experiment completed in {experiment_duration:.1f}s")
        return summary
    
    def analyze_conversation(self, transcript: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze conversation with comprehensive metrics."""
        if not transcript:
            return {"error": "Empty transcript"}
        
        # Basic statistics
        total_messages = len(transcript)
        total_words = sum(len(msg.get("text", "").split()) for msg in transcript)
        
        # Participation analysis
        speaker_stats = defaultdict(lambda: {"messages": 0, "words": 0})
        role_stats = defaultdict(lambda: {"messages": 0, "words": 0})
        
        for msg in transcript:
            speaker = msg.get("speaker", "Unknown")
            role = msg.get("role", "Unknown")
            words = len(msg.get("text", "").split())
            
            speaker_stats[speaker]["messages"] += 1
            speaker_stats[speaker]["words"] += words
            
            role_stats[role]["messages"] += 1
            role_stats[role]["words"] += words
        
        # Interaction patterns
        response_patterns = defaultdict(lambda: defaultdict(int))
        for i in range(1, len(transcript)):
            prev_speaker = transcript[i-1].get("speaker", "Unknown")
            curr_speaker = transcript[i].get("speaker", "Unknown")
            response_patterns[prev_speaker][curr_speaker] += 1
        
        # Content analysis - simple keyword tracking
        tech_keywords = {"system", "architecture", "api", "performance", "scalability"}
        ux_keywords = {"user", "experience", "interface", "design", "usability"}
        business_keywords = {"market", "revenue", "cost", "strategy", "value"}
        
        def count_keywords(text: str, keywords: set) -> int:
            return sum(1 for word in text.lower().split() if word in keywords)
        
        content_focus = {
            "technical": sum(count_keywords(msg.get("text", ""), tech_keywords) for msg in transcript),
            "ux": sum(count_keywords(msg.get("text", ""), ux_keywords) for msg in transcript),
            "business": sum(count_keywords(msg.get("text", ""), business_keywords) for msg in transcript)
        }
        
        # Turn analysis
        turns_data = defaultdict(list)
        for msg in transcript:
            turns_data[msg.get("turn", 1)].append(msg)
        
        turn_stats = {
            "total_turns": len(turns_data),
            "avg_messages_per_turn": total_messages / len(turns_data) if turns_data else 0,
            "turn_lengths": {turn: len(msgs) for turn, msgs in turns_data.items()}
        }
        
        return {
            "overview": {
                "total_messages": total_messages,
                "total_words": total_words,
                "unique_speakers": len(speaker_stats),
                "avg_message_length": total_words / total_messages if total_messages > 0 else 0
            },
            "participation": {
                "by_speaker": dict(speaker_stats),
                "by_role": dict(role_stats)
            },
            "interaction_patterns": {
                "response_matrix": {k: dict(v) for k, v in response_patterns.items()},
                "most_active_pairs": sorted(
                    [(f"{s1}->{s2}", count) for s1, responses in response_patterns.items() 
                     for s2, count in responses.items()],
                    key=lambda x: x[1], reverse=True
                )[:5]
            },
            "content_analysis": {
                "focus_areas": content_focus,
                "dominant_focus": max(content_focus.keys(), key=lambda k: content_focus[k])
            },
            "turn_analysis": turn_stats
        }
    
    def save_results(self, results: Dict[str, Any], output_dir: Optional[Path] = None) -> Dict[str, Path]:
        """Save simulation results in multiple formats with comprehensive reporting."""
        
        if output_dir is None:
            output_dir = Path(self.config.output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = {}
        
        # Save transcript as JSONL
        if "transcript" in results:
            jsonl_path = output_dir / f"conversation_log_{timestamp}.jsonl"
            with jsonl_path.open("w", encoding="utf-8") as f:
                for msg in results["transcript"]:
                    f.write(json.dumps(msg, ensure_ascii=False) + "\n")
            saved_files["jsonl"] = jsonl_path
        
        # Save full results as JSON
        json_path = output_dir / f"simulation_results_{timestamp}.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        saved_files["json"] = json_path
        
        # Generate analysis if enabled
        if self.config.auto_analysis and "transcript" in results:
            analysis = self.analyze_conversation(results["transcript"])
            
            # Save analysis
            analysis_path = output_dir / f"analysis_{timestamp}.json"
            with analysis_path.open("w", encoding="utf-8") as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
            saved_files["analysis"] = analysis_path
            
            # Generate markdown report
            report_path = self._generate_markdown_report(results, analysis, output_dir, timestamp)
            saved_files["report"] = report_path
        
        return saved_files
    
    def _generate_markdown_report(self, results: Dict[str, Any], analysis: Dict[str, Any], 
                                 output_dir: Path, timestamp: str) -> Path:
        """Generate a comprehensive markdown report."""
        
        report_path = output_dir / f"simulation_report_{timestamp}.md"
        
        lines = [
            f"# Simulation Report - {timestamp}",
            "",
            "## Configuration",
            f"- **Model**: {results['config']['model']}",
            f"- **Temperature**: {results['config']['temperature']}",
            f"- **Turns**: {results['config']['turns']}",
            f"- **Seed**: {results['config']['seed']}",
            "",
            "## Performance Metrics",
            f"- **Duration**: {results['metrics']['duration']:.1f} seconds",
            f"- **Total Messages**: {results['metrics']['total_messages']}",
            f"- **Total Tokens**: {results['metrics']['total_tokens']:,}",
            f"- **API Calls**: {results['metrics']['api_calls']}",
            f"- **Errors**: {results['metrics']['errors']}",
            f"- **Average Response Time**: {results['metrics']['average_response_time']:.2f}s",
            f"- **Messages per Minute**: {results['metrics']['messages_per_minute']:.1f}",
            "",
            "## Participants",
        ]
        
        for agent in results.get("agents", []):
            lines.append(f"- **{agent['name']}** ({agent['role']})")
        
        lines.extend([
            "",
            "## Analysis Overview",
            f"- **Total Words**: {analysis['overview']['total_words']:,}",
            f"- **Unique Speakers**: {analysis['overview']['unique_speakers']}",
            f"- **Average Message Length**: {analysis['overview']['avg_message_length']:.1f} words",
            f"- **Dominant Content Focus**: {analysis['content_analysis']['dominant_focus'].title()}",
            "",
            "## Participation Distribution",
            "| Speaker | Messages | Words | Avg Length |",
            "|---------|----------|-------|------------|",
        ])
        
        for speaker, stats in analysis["participation"]["by_speaker"].items():
            avg_length = stats["words"] / stats["messages"] if stats["messages"] > 0 else 0
            lines.append(f"| {speaker} | {stats['messages']} | {stats['words']} | {avg_length:.1f} |")
        
        lines.extend([
            "",
            "## Content Focus Distribution",
            "| Focus Area | Mentions |",
            "|------------|----------|",
        ])
        
        for focus, count in analysis["content_analysis"]["focus_areas"].items():
            lines.append(f"| {focus.title()} | {count} |")
        
        lines.extend([
            "",
            "## Top Interaction Patterns",
            "| Pattern | Frequency |",
            "|---------|-----------|",
        ])
        
        for pattern, count in analysis["interaction_patterns"]["most_active_pairs"]:
            lines.append(f"| {pattern} | {count} |")
        
        # Add transcript
        lines.extend([
            "",
            "## Transcript",
            ""
        ])
        
        for msg in results.get("transcript", []):
            lines.extend([
                f"**Turn {msg['turn']} - {msg['speaker']} ({msg['role']})**",
                msg['text'],
                ""
            ])
        
        # Add error log if any
        if hasattr(self, "error_log") and self.error_log:
            lines.extend([
                "",
                "## Error Log",
                ""
            ])
            for error in self.error_log:
                lines.extend([
                    f"- **{error['timestamp']}**: {error['agent']} (Attempt {error['attempt']}) - {error['error']}",
                ])
        
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return report_path
    
    def cleanup(self):
        """Clean up resources."""
        if self.worker_pool:
            self.worker_pool.shutdown(wait=True)

# ==================== COMMAND LINE INTERFACE ====================

def load_config_from_file(config_file: Path) -> SimulationConfig:
    """Load configuration from INI file."""
    if not config_file.exists():
        return SimulationConfig()
    
    parser = configparser.ConfigParser()
    parser.read(config_file)
    
    config = SimulationConfig()
    
    # Map INI sections to config attributes
    if parser.has_section("OpenAI"):
        if parser.has_option("OpenAI", "model"):
            config.model = parser.get("OpenAI", "model")
        if parser.has_option("OpenAI", "temperature"):
            config.temperature = parser.getfloat("OpenAI", "temperature")
    
    if parser.has_section("Simulation"):
        if parser.has_option("Simulation", "parallel_agent_actions"):
            config.parallel_execution = parser.getboolean("Simulation", "parallel_agent_actions")
    
    if parser.has_section("run"):
        if parser.has_option("run", "max_turns"):
            config.max_turns = parser.getint("run", "max_turns")
        if parser.has_option("run", "seed"):
            config.seed = parser.get("run", "seed")
    
    return config

def main():
    parser = argparse.ArgumentParser(
        description="TinyTroupe Simulation Engine - Production-ready persona simulation tool"
    )
    
    # Core simulation parameters
    parser.add_argument("--mode", choices=["single", "experiment", "validate"], 
                       default="single", help="Simulation mode")
    parser.add_argument("--seed", type=str, help="Discussion topic/prompt")
    parser.add_argument("--turns", type=int, help="Number of conversation turns")
    parser.add_argument("--personas", nargs="+", help="Specific personas to include")
    
    # Model configuration
    parser.add_argument("--model", type=str, help="Language model to use")
    parser.add_argument("--temperature", type=float, help="Model temperature")
    
    # Experiment mode
    parser.add_argument("--variants", nargs="+", help="Experiment variants to run")
    parser.add_argument("--runs", type=int, default=1, help="Runs per variant")
    
    # Performance options
    parser.add_argument("--parallel", action="store_true", help="Enable parallel execution")
    parser.add_argument("--max-workers", type=int, default=3, help="Maximum worker threads")
    
    # File paths
    parser.add_argument("--config", type=Path, help="Configuration file path")
    parser.add_argument("--personas-file", type=Path, help="Personas JSON file")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    
    # Validation and monitoring
    parser.add_argument("--validate-only", action="store_true", help="Only validate personas")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Load configuration
    config_file = args.config or Path("config/config.ini")
    config = load_config_from_file(config_file) if config_file.exists() else SimulationConfig()
    
    # Override config with command line arguments
    if args.seed:
        config.seed = args.seed
    if args.turns:
        config.max_turns = args.turns
    if args.model:
        config.model = args.model
    if args.temperature:
        config.temperature = args.temperature
    if args.personas_file:
        config.personas_file = str(args.personas_file)
    if args.output_dir:
        config.output_dir = str(args.output_dir)
    if args.parallel:
        config.parallel_execution = True
    if args.max_workers:
        config.max_workers = args.max_workers
    
    config.log_level = args.log_level
    
    # Create engine
    engine = SimulationEngine(config)
    
    try:
        if args.mode == "validate" or args.validate_only:
            # Validate personas only
            personas = engine.persona_db.load_personas()
            print(f"✓ Successfully validated {len(personas)} personas")
            
            for persona in personas:
                issues = persona.validate()
                if issues:
                    print(f"⚠ {persona.name}: {', '.join(issues)}")
                else:
                    print(f"✓ {persona.name}: Valid")
        
        elif args.mode == "single":
            # Single simulation
            if not config.seed:
                parser.error("Seed prompt is required for single simulation mode")
            
            print(f"Running single simulation: {config.seed}")
            results = engine.run_simulation(
                seed=config.seed,
                turns=config.max_turns,
                personas=args.personas
            )
            
            if results["success"]:
                saved_files = engine.save_results(results)
                print(f"✓ Simulation completed successfully")
                print(f"Results saved to: {list(saved_files.values())}")
            else:
                print(f"✗ Simulation failed: {results.get('error', 'Unknown error')}")
        
        elif args.mode == "experiment":
            # Experiment mode - load variants from config or use defaults
            if args.variants:
                # Simple variant creation from command line
                variants = {}
                for variant_name in args.variants:
                    variants[variant_name] = {
                        "seed": config.seed,
                        "temperature": config.temperature + (0.1 if variant_name == "creative" else 0),
                    }
            else:
                # Default variants
                variants = {
                    "baseline": {
                        "seed": config.seed,
                        "temperature": config.temperature,
                    },
                    "creative": {
                        "seed": config.seed,
                        "temperature": min(config.temperature + 0.3, 1.5),
                    }
                }
            
            print(f"Running experiment with {len(variants)} variants, {args.runs} runs each")
            
            experiment_results = engine.run_experiments(variants, args.runs)
            
            # Save experiment results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(config.output_dir) / "experiments"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            results_file = output_dir / f"experiment_results_{timestamp}.json"
            with results_file.open("w", encoding="utf-8") as f:
                json.dump(experiment_results, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"✓ Experiment completed in {experiment_results['experiment_duration']:.1f}s")
            print(f"Results saved to: {results_file}")
    
    finally:
        engine.cleanup()

if __name__ == "__main__":
    main()