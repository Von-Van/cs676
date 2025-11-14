# TinyTroupe Simulation Engine - Complete Documentation

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Setup & Installation](#setup--installation)
4. [Configuration](#configuration)
5. [Persona Database](#persona-database)
6. [Usage Guide](#usage-guide)
7. [API Reference](#api-reference)
8. [Monitoring & Performance](#monitoring--performance)
9. [Troubleshooting](#troubleshooting)
10. [Scaling & Production](#scaling--production)
11. [Maintenance](#maintenance)

## Overview

The TinyTroupe Simulation Engine is a production-ready system for conducting scalable, monitored persona-based conversations. It consolidates all previous script functionality into a unified platform with comprehensive validation, error handling, and performance monitoring.

### Key Features

- **🏗️ Unified Architecture**: Single script handles all simulation modes
- **🔍 Real-time Monitoring**: Performance tracking and error handling  
- **📊 Automatic Analysis**: Built-in conversation analytics
- **🛡️ Robust Validation**: Persona and configuration validation
- **⚡ Scalable Design**: Parallel execution and load balancing
- **📈 Production Ready**: Rate limiting, graceful degradation, monitoring

### What's New vs Previous Scripts

| Previous | Consolidated Engine |
|----------|-------------------|
| Multiple scripts (run_simulation, analyze_conversations, export_readme, run_manyexperiment) | Single `simulation_engine.py` |
| Basic error handling | Comprehensive error recovery and monitoring |
| Manual analysis | Automatic analysis with visualization |
| Limited validation | Full persona and config validation |
| No rate limiting | Built-in rate limiting and cost controls |
| Basic export | Multiple export formats with analysis |

## System Architecture

```
TinyTroupe Simulation Engine
├── Core Engine (simulation_engine.py)
│   ├── SimulationEngine: Main orchestrator
│   ├── PersonaDatabase: Persona management
│   ├── SimulationConfig: Configuration management
│   └── SimulationMetrics: Performance tracking
├── UI Layer (app_updated.py)
│   ├── Streamlit interface
│   ├── Real-time monitoring
│   └── Interactive controls
├── Data Layer
│   ├── personas.agents.json: Persona database
│   ├── config.ini: System configuration
│   └── outputs/: Results and analytics
└── Monitoring
    ├── Performance metrics
    ├── Error logging
    └── Rate limiting
```

### Component Interaction Flow

1. **Configuration Loading**: System loads config from multiple sources
2. **Persona Validation**: Database validates all persona specifications
3. **Simulation Execution**: Engine orchestrates conversation with monitoring
4. **Real-time Analysis**: Automatic analysis during and after simulation
5. **Result Export**: Multiple format export with comprehensive reporting

## Setup & Installation

### Prerequisites

```bash
# Core dependencies
pip install streamlit
pip install tinytroupe
pip install pandas numpy  # For analysis features
pip install matplotlib seaborn  # For visualizations (optional)
```

### Directory Structure

Ensure your project has this structure:

```
project/
├── scripts/
│   └── simulation_engine.py
├── data/
│   └── personas.agents.json
├── config/
│   └── config.ini
├── outputs/
│   ├── conversations/
│   ├── analysis/
│   └── experiments/
├── app_updated.py
└── requirements.txt
```

### Environment Setup

1. **API Key Configuration** (Priority Order):
   ```bash
   # Method 1: Streamlit secrets (recommended)
   echo 'api_key = "your-openai-key"' > .streamlit/secrets.toml
   
   # Method 2: Environment variable
   export OPENAI_API_KEY="your-openai-key"
   
   # Method 3: Config file (less secure)
   # Add to config/config.ini under [openai] section
   ```

2. **Run Application**:
   ```bash
   streamlit run app_updated.py
   ```

## Configuration

### SimulationConfig Options

```python
@dataclass
class SimulationConfig:
    # Core simulation settings
    model: str = "gpt-4o-mini"           # LLM model
    temperature: float = 0.7              # Response creativity
    max_turns: int = 6                    # Conversation rounds
    seed: str = ""                        # Discussion topic
    
    # Persona settings  
    personas_file: str = "personas.agents.json"
    max_personas: int = 10                # Performance limit
    persona_timeout: float = 30.0         # Individual timeout
    
    # Performance & scaling
    parallel_execution: bool = False      # Enable parallel processing
    max_workers: int = 3                  # Thread pool size
    rate_limit_delay: float = 1.0         # Delay between API calls
    max_retries: int = 5                  # Retry attempts
    batch_size: int = 5                   # Batch processing size
    
    # Monitoring & logging
    enable_monitoring: bool = True        # Performance tracking
    log_level: str = "INFO"               # Logging verbosity
    performance_tracking: bool = True     # Detailed metrics
    
    # Output settings
    output_dir: str = "outputs"           # Results directory
    auto_analysis: bool = True            # Automatic analysis
    save_visualizations: bool = True      # Generate charts
    
    # Error handling
    timeout_seconds: int = 300            # Overall timeout
    graceful_degradation: bool = True     # Continue on errors
    fallback_model: str = "gpt-3.5-turbo" # Backup model
```

### INI File Configuration

```ini
# config/config.ini
[OpenAI]
model = gpt-4o-mini
temperature = 0.7
max_tokens = 512
timeout = 60
max_attempts = 3
exponential_backoff_factor = 2

[Simulation]
parallel_agent_generation = True
parallel_agent_actions = True
max_turns = 6

[Performance] 
rate_limit_delay = 1.0
max_workers = 3
batch_size = 5

[Monitoring]
enable_monitoring = True
log_level = INFO
performance_tracking = True
```

## Persona Database

### Enhanced Persona Schema

```json
{
  "type": "TinyPerson",
  "id": "unique_identifier",
  "name": "Display Name",
  "role": "Professional Role",
  "description": "Background and expertise",
  "personality": {
    "traits": ["trait1", "trait2", "trait3"],
    "goals": ["goal1", "goal2"],
    "style": "Communication style",
    "communication": "structured|visual|direct",
    "decisions": "data-driven|intuitive|collaborative", 
    "approach": "systematic|creative|pragmatic"
  },
  "model": "gpt-4o-mini",
  "temperature": 0.4,
  "prompting": {
    "format": "markdown|text",
    "role": "System role definition"
  },
  "validation_rules": {
    "required_traits": ["analytical"],
    "max_goals": 5,
    "style_constraints": ["professional"]
  }
}
```

### Persona Validation Rules

The system validates each persona against these criteria:

- **Required Fields**: `id`, `name`, `role`, `description`, `personality`
- **ID Format**: Alphanumeric, underscore, dash only
- **Temperature Range**: 0.0 to 2.0
- **Personality Structure**: Must contain `traits`, `goals`, `style`
- **Uniqueness**: No duplicate IDs or names
- **Completeness**: All personality fields populated

### Database Management

```python
# Loading personas with validation
persona_db = PersonaDatabase(config)
personas = persona_db.load_personas("data/personas.agents.json")

# Validation
for persona in personas:
    issues = persona.validate()
    if issues:
        print(f"Validation issues for {persona.name}: {issues}")

# Saving with backup
persona_db.save_personas(personas, output_file)
```

## Usage Guide

### Command Line Interface

```bash
# Single simulation
python scripts/simulation_engine.py \
    --mode single \
    --seed "Evaluate AI-powered content moderation" \
    --turns 6 \
    --model gpt-4o-mini

# Experiment mode
python scripts/simulation_engine.py \
    --mode experiment \
    --variants baseline creative \
    --runs 3 \
    --parallel

# Validation only
python scripts/simulation_engine.py \
    --mode validate \
    --personas-file data/personas.agents.json
```

### Programmatic Usage

```python
from scripts.simulation_engine import SimulationEngine, SimulationConfig

# Configure engine
config = SimulationConfig(
    model="gpt-4o-mini",
    temperature=0.7,
    max_turns=6,
    enable_monitoring=True
)

engine = SimulationEngine(config)

# Run simulation
results = engine.run_simulation(
    seed="Discuss AI ethics in content moderation",
    turns=6,
    personas=["Dr. Sarah Chen", "Marcus Rivera"]
)

# Analysis
if results['success']:
    analysis = engine.analyze_conversation(results['transcript'])
    print(f"Analyzed {analysis['overview']['total_messages']} messages")
    
    # Save results
    saved_files = engine.save_results(results)
    print(f"Results saved to {saved_files}")
```

### Streamlit Interface

1. **Configure Parameters**: Use sidebar controls
2. **Select Personas**: Choose from validated personas
3. **Create Custom Personas**: Use the persona builder
4. **Set Environment**: Configure discussion context
5. **Run Simulation**: Monitor real-time progress
6. **View Results**: Interactive transcript and analysis
7. **Export**: Multiple format download options

## API Reference

### SimulationEngine Class

```python
class SimulationEngine:
    def __init__(self, config: SimulationConfig)
    
    def run_simulation(
        self, 
        seed: str, 
        turns: Optional[int] = None,
        personas: Optional[List[str]] = None,
        custom_personas: Optional[List[PersonaSpec]] = None
    ) -> Dict[str, Any]
    
    def run_experiments(
        self, 
        variants: Dict[str, Dict[str, Any]], 
        runs_per_variant: int = 1
    ) -> Dict[str, Any]
    
    def analyze_conversation(
        self, 
        transcript: List[Dict[str, Any]]
    ) -> Dict[str, Any]
    
    def save_results(
        self, 
        results: Dict[str, Any], 
        output_dir: Optional[Path] = None
    ) -> Dict[str, Path]
```

### PersonaDatabase Class

```python
class PersonaDatabase:
    def load_personas(
        self, 
        personas_file: Optional[Path] = None
    ) -> List[PersonaSpec]
    
    def create_tiny_persons(
        self, 
        persona_specs: List[PersonaSpec]
    ) -> List[TinyPerson]
    
    def save_personas(
        self, 
        personas: List[PersonaSpec], 
        output_file: Path
    )
```

### Results Format

```python
{
    "success": True,
    "transcript": [
        {
            "turn": 1,
            "speaker": "Dr. Sarah Chen",
            "role": "Systems Architect", 
            "text": "Message content...",
            "timestamp": "2024-01-01T10:00:00Z"
        }
    ],
    "agents": [
        {
            "name": "Dr. Sarah Chen",
            "role": "Systems Architect"
        }
    ],
    "metrics": {
        "duration": 45.2,
        "total_messages": 12,
        "total_tokens": 2500,
        "api_calls": 15,
        "errors": 0,
        "average_response_time": 2.1
    },
    "config": {
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "turns": 6
    }
}
```

## Monitoring & Performance

### Performance Metrics

The system tracks comprehensive performance metrics:

- **Duration**: Total simulation time
- **Throughput**: Messages per minute
- **API Efficiency**: Calls per message
- **Response Time**: Average API response time
- **Error Rate**: Failed requests percentage
- **Token Usage**: Token consumption tracking

### Real-time Monitoring

```python
# Access metrics during simulation
engine = SimulationEngine(config)
results = engine.run_simulation(seed, turns)

print(f"Duration: {results['metrics']['duration']:.1f}s")
print(f"Messages: {results['metrics']['total_messages']}")
print(f"API Calls: {results['metrics']['api_calls']}")
print(f"Error Rate: {results['metrics']['errors']}/{results['metrics']['api_calls']}")
```

### Cost Tracking

```python
# Estimate costs before running
def estimate_cost(num_personas: int, turns: int, model: str = "gpt-4o-mini"):
    tokens_per_turn = num_personas * 500  # Rough estimate
    total_tokens = tokens_per_turn * turns
    
    # Model pricing (tokens per dollar)
    pricing = {
        "gpt-4o-mini": 0.002 / 1000,  # $0.002 per 1K tokens
        "gpt-3.5-turbo": 0.001 / 1000,
        "gpt-4o": 0.03 / 1000
    }
    
    return total_tokens * pricing.get(model, 0.002 / 1000)

estimated_cost = estimate_cost(5, 6)  # 5 personas, 6 turns
print(f"Estimated cost: ${estimated_cost:.4f}")
```

## Troubleshooting

### Common Issues

#### 1. Persona Validation Failures

```
Error: Persona validation errors: Missing required field: description
```

**Solution**: Ensure all personas have required fields:
```python
# Check persona completeness
for persona in personas:
    issues = persona.validate()
    if issues:
        print(f"Fix {persona.name}: {issues}")
```

#### 2. Rate Limiting

```
Error: 429 Rate limit exceeded
```

**Solutions**:
- Increase `rate_limit_delay` in config
- Enable API caching
- Use smaller batch sizes
- Implement exponential backoff

```python
config = SimulationConfig(
    rate_limit_delay=2.0,  # Increase delay
    max_retries=5,         # More retry attempts
    parallel_execution=False  # Disable parallel for rate-limited APIs
)
```

#### 3. Memory Issues with Large Simulations

```
Error: Out of memory during simulation
```

**Solutions**:
- Reduce `max_personas` limit
- Lower `max_turns`
- Enable `graceful_degradation`
- Use streaming processing

```python
config = SimulationConfig(
    max_personas=5,           # Limit concurrent personas
    batch_size=3,             # Process in smaller batches
    graceful_degradation=True # Continue on errors
)
```

#### 4. API Key Configuration

```
Error: OpenAI API key not found
```

**Solution**: Set API key using priority order:
1. Streamlit secrets: `.streamlit/secrets.toml`
2. Environment variable: `OPENAI_API_KEY`
3. Config file: `config/config.ini`

#### 5. TinyTroupe Library Issues

```
Error: 'Document' object has no attribute 'text'
```

**Solution**: The engine handles this automatically with `graceful_degradation`, but you can also:
- Update TinyTroupe to latest version
- Enable memory consolidation workarounds
- Use fallback text extraction

### Debug Mode

Enable debug logging for detailed troubleshooting:

```python
config = SimulationConfig(
    log_level="DEBUG",
    enable_monitoring=True,
    performance_tracking=True
)

# Or in Streamlit
st.checkbox("Show Debug Info", value=True)
```

### Performance Optimization

#### For High-Volume Simulations

```python
config = SimulationConfig(
    parallel_execution=True,
    max_workers=5,
    batch_size=10,
    rate_limit_delay=0.5,
    enable_monitoring=True
)
```

#### For Cost Optimization

```python
config = SimulationConfig(
    model="gpt-4o-mini",      # Cheaper model
    temperature=0.5,          # More predictable (fewer retries)
    max_personas=3,           # Fewer concurrent agents
    rate_limit_delay=1.5      # Avoid rate limits
)
```

## Scaling & Production

### Production Deployment Checklist

- [ ] **Security**: API keys in secure storage (not config files)
- [ ] **Rate Limiting**: Configured for your API tier
- [ ] **Monitoring**: Error tracking and alerting setup  
- [ ] **Backup**: Persona database backup strategy
- [ ] **Logging**: Centralized logging system
- [ ] **Resources**: Sufficient memory and CPU for concurrent simulations

### Load Balancing

For high-volume deployments:

```python
# Multiple engine instances
configs = [
    SimulationConfig(max_workers=2, rate_limit_delay=1.0),
    SimulationConfig(max_workers=2, rate_limit_delay=1.2), 
    SimulationConfig(max_workers=2, rate_limit_delay=1.4)
]

engines = [SimulationEngine(config) for config in configs]

# Round-robin assignment
def get_next_engine():
    return engines[request_count % len(engines)]
```

### Monitoring in Production

```python
import logging
import time
from datetime import datetime

# Production logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simulation_engine.log'),
        logging.StreamHandler()
    ]
)

# Performance monitoring
def monitor_simulation(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            # Log success metrics
            logging.info(f"Simulation completed in {duration:.2f}s")
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Log error metrics
            logging.error(f"Simulation failed after {duration:.2f}s: {e}")
            raise
    
    return wrapper
```

### Database Scaling

For larger persona databases:

```python
# Lazy loading for large persona sets
class ScalablePersonaDatabase(PersonaDatabase):
    def __init__(self, config):
        super().__init__(config)
        self._persona_index = {}
        self._load_index()
    
    def _load_index(self):
        """Load persona metadata without full content"""
        # Implementation for indexing
        pass
    
    def load_personas_by_tags(self, tags: List[str]) -> List[PersonaSpec]:
        """Load only personas matching specific tags"""
        # Implementation for filtered loading
        pass
```

## Maintenance

### Regular Maintenance Tasks

#### Daily
- Monitor error logs for patterns
- Check API usage against limits
- Validate persona database integrity

#### Weekly  
- Update persona database backups
- Review performance metrics
- Clean old output files

#### Monthly
- Update TinyTroupe library
- Review and optimize configurations
- Update documentation

### Backup Strategy

```python
# Automated backup script
import shutil
from datetime import datetime

def backup_personas():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path("backups") / timestamp
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Backup persona database
    shutil.copy2("data/personas.agents.json", 
                 backup_dir / "personas.agents.json")
    
    # Backup config
    shutil.copy2("config/config.ini", 
                 backup_dir / "config.ini")
    
    print(f"Backup created: {backup_dir}")

# Schedule weekly backups
backup_personas()
```

### Health Checks

```python
def system_health_check():
    """Comprehensive system health validation"""
    issues = []
    
    # Check persona database
    try:
        engine = SimulationEngine(SimulationConfig())
        personas = engine.persona_db.load_personas()
        print(f"✓ Loaded {len(personas)} personas")
    except Exception as e:
        issues.append(f"Persona loading failed: {e}")
    
    # Check API connectivity
    try:
        # Simple test simulation
        test_results = engine.run_simulation(
            seed="Test connectivity", 
            turns=1,
            personas=personas[:1] if personas else None
        )
        if test_results.get('success'):
            print("✓ API connectivity working")
        else:
            issues.append("API test failed")
    except Exception as e:
        issues.append(f"API connectivity failed: {e}")
    
    # Check disk space
    import shutil
    free_space = shutil.disk_usage(".").free / (1024**3)  # GB
    if free_space < 1:  # Less than 1GB
        issues.append(f"Low disk space: {free_space:.1f}GB")
    else:
        print(f"✓ Disk space: {free_space:.1f}GB available")
    
    return issues

# Run health check
issues = system_health_check()
if issues:
    print("⚠️ Health check issues:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("✅ System healthy")
```

### Version Management

```python
# Version tracking
__version__ = "2.0.0"
__compatible_tinytroupe__ = ">=0.3.0"
__last_updated__ = "2024-01-01"

def check_compatibility():
    """Verify system component compatibility"""
    try:
        import tinytroupe
        print(f"TinyTroupe version: {tinytroupe.__version__}")
        print(f"Engine version: {__version__}")
        print(f"Compatible: {__compatible_tinytroupe__}")
    except ImportError:
        print("⚠️ TinyTroupe not installed")
```

---

## Summary

This comprehensive documentation covers all aspects of the consolidated TinyTroupe Simulation Engine. The system provides:

1. **Unified Architecture**: Single script replaces multiple tools
2. **Production Features**: Monitoring, validation, error handling
3. **Scalability**: Parallel execution, load balancing, cost optimization  
4. **Ease of Use**: Streamlit interface with real-time feedback
5. **Comprehensive Analysis**: Automatic conversation analytics
6. **Maintainability**: Health checks, backups, version management

The engine is designed to handle both small-scale experimentation and large-scale production deployments while maintaining reliability and performance.