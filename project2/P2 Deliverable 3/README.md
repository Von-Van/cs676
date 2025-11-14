---
title: AI Persona Panel Simulator
emoji: 🧠
colorFrom: purple
colorTo: blue
sdk: streamlit
sdk_version: 1.32.0
app_file: app_updated.py
pinned: false
license: mit
---

# 🧠 AI Persona Panel Simulator

A production-ready multi-agent simulation platform built on Microsoft TinyTroupe with advanced features including agent reuse, validation, monitoring, security, and comprehensive analysis.

## ⚙️ Hugging Face Configuration

**Important**: This app requires an OpenAI API key to function.

### Add Your API Key

1. Go to your Space **Settings** tab
2. Scroll to **Repository secrets**
3. Click **New secret**
4. Add:
   - **Name**: `OPENAI_API_KEY`
   - **Value**: Your OpenAI API key (starts with `sk-`)
5. Click **Add secret**
6. The Space will restart automatically

### Optional Configuration

You can also configure these settings in the sidebar:
- **Model**: Choose between GPT-4o-mini (faster, cheaper) or GPT-4 (more capable)
- **Temperature**: Adjust creativity (0.0-1.0)
- **Max Rounds**: Number of conversation turns

## 🚀 Quick Start

### For Hugging Face Users
1. Add your OpenAI API key as a secret (see above)
2. Wait for the Space to load
3. Enter a discussion topic
4. Select personas (manual or AI auto-selection)
5. Click "Run Simulation"

### For Local Installation
**Prerequisites**: Python 3.10+, OpenAI API key

### Installation

1. **Clone or download the project**
   ```bash
   cd "project2/P2 Deliverable 3"
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure OpenAI API**
   
   Edit `config/config.ini` and add your API key:
   ```ini
   [OpenAI]
   api_key = sk-your-api-key-here
   model = gpt-4o-mini
   ```

4. **Launch the application**
   
   **Windows (Batch File)**:
   ```bash
   launch_app.bat
   ```
   
   **PowerShell**:
   ```powershell
   .\launch_app.ps1
   ```
   
   **Manual**:
   ```bash
   streamlit run app_updated.py
   ```

5. **Access the interface**
   - Open browser to `http://localhost:8501`
   - Start creating personas and running simulations!

---

## ✨ Key Features

### 🤖 Agent Management
- **Agent Reuse**: Use agents across multiple conversations
- **Individual & Group Chat**: Ask follow-up questions to agents individually or as a group
- **Unique Agent Lifecycle**: Automatic name collision prevention with timestamp-based unique IDs
- **Persistent Agent Memory**: Agents maintain conversation history across sessions

### ✅ Persona Validation System
Seven validation types ensure persona consistency and realism:
1. **Trait Consistency**: Detects contradictions in agent characteristics
2. **Role Alignment**: Validates role-description compatibility
3. **Realism Check**: Flags unrealistic combinations and suggests improvements
4. **Depth Analysis**: Ensures goals, background, and motivations are present
5. **Tone Consistency**: Validates communication style matches persona
6. **Conversation Consistency**: Checks persona behavior in simulated conversations
7. **Batch Validation**: Validates multiple personas simultaneously

Validation provides:
- Overall consistency score (0-100)
- Realism score (0-100)
- Specific warnings for issues
- Actionable suggestions for improvements

### 🎯 Conversation Features
- **AI Persona Selection**: Choose from 50+ pre-defined personas or create custom ones
- **Real-time Validation**: Custom personas validated during creation with immediate feedback
- **Multi-turn Conversations**: Configure conversation length (1-10 turns)
- **Diverse Topics**: 100+ conversation topics across categories
- **Rich Context**: Personas engage with realistic backgrounds and motivations

### 🔍 Analysis & Insights
- **AI Thoughts Tracking**: Capture agent reasoning from episodic memory
- **Agreement Analysis**: Track consensus patterns across conversations
- **Sentiment Analysis**: Emotional tone tracking for each turn
- **Topic Evolution**: Visualize how topics develop over conversation
- **Participation Metrics**: Measure engagement and contribution balance
- **Conversation Flow**: Graph-based visualization of dialogue structure
- **Export Capabilities**: JSON, JSONL, and Markdown formats

### 📊 Visualization Suite
- **Agreement Heatmaps**: Visualize consensus patterns
- **Sentiment Timelines**: Track emotional progression
- **Participation Charts**: Analyze contribution balance
- **Topic Word Clouds**: Identify key themes
- **Conversation Graphs**: Network analysis of dialogue flow

### 🛡️ Production Features

#### Security System (`scripts/security_manager.py`)
- **Optional Authentication**: Configurable login system
- **Password Hashing**: SHA-256 with salt
- **Session Management**: Automatic timeout and cleanup
- **Rate Limiting**: Per-user and per-IP protection
- **Audit Logging**: Track all authentication events
- **Input Sanitization**: Prevent injection attacks

#### Backup System (`scripts/backup_manager.py`)
- **Automatic Backups**: Configurable interval (default: 60 minutes)
- **Manual Backups**: On-demand backup creation
- **Compression**: Reduced storage with gzip
- **Integrity Checking**: SHA-256 checksums for verification
- **Backup Rotation**: Automatic cleanup based on age and count
- **Safe Restore**: Creates safety backup before restore

#### Monitoring System (`scripts/monitoring.py`)
- **System Metrics**: CPU, memory, disk usage tracking
- **Performance Metrics**: Request rate, response time, error rate
- **Health Checks**: Automatic status determination (healthy/degraded/unhealthy)
- **Alert Thresholds**: Configurable warning levels
- **Prometheus Export**: `/metrics` endpoint for integration
- **Time-Series Data**: Historical metric tracking

#### Error Handling & Resilience
- **Retry Mechanisms**: Automatic retry with exponential backoff (max: 5 attempts)
- **Rate Limit Handling**: Special backoff for API rate limits
- **Timeout Management**: Separate tracking for timeout errors
- **Graceful Degradation**: Continue with partial failures
- **Circuit Breaker**: Prevent cascade failures

---

## 📖 Usage Guide

### Creating and Running Simulations

1. **Select Conversation Topic**
   - Choose from 100+ topics in categories: Technology, Social, Business, etc.
   - Or enter a custom topic

2. **Configure Simulation**
   - Set number of conversation turns (1-10)
   - Optionally enable parallel execution for better performance

3. **Choose Personas**
   - **AI Selection**: System automatically selects 3-5 diverse, suitable personas
   - **Manual Selection**: Choose from 50+ pre-defined personas
   - **Custom Creation**: Design your own personas with real-time validation

4. **Run Simulation**
   - Click "Run Simulation" button
   - Monitor progress in real-time
   - View results across 6 tabs

### Using Agent Chat

After running a simulation:

1. Navigate to **Agent Chat** tab
2. Choose mode:
   - **🌐 All Agents (Group Chat)**: Ask questions to all agents simultaneously
   - **Individual Agent**: Select specific agent for one-on-one conversation
3. Type your question and press Enter
4. View responses with agent thoughts and reasoning

### Creating Custom Personas

1. Click **"📝 Create Custom Persona"** in sidebar
2. Fill in persona details:
   - Name and role
   - Characteristics (comma-separated traits)
   - Description
   - Goals
   - Communication style
3. See validation scores and warnings in real-time
4. Click "Add Custom Persona" when satisfied
5. Persona appears in manual selection list

### Exporting Results

1. Navigate to **Export** tab
2. Choose format:
   - **JSON**: Full structured data with metadata
   - **JSONL**: Streaming format, one conversation per line
   - **Markdown**: Human-readable conversation log
3. Click "Download" button

---

## ⚙️ Configuration

### OpenAI Settings (`config/config.ini`)

```ini
[OpenAI]
api_key = sk-your-key-here
model = gpt-4o-mini          # Or gpt-4, gpt-3.5-turbo
temperature = 0.7            # Creativity level (0.0-2.0)
max_tokens = 4096            # Response length limit
timeout = 60                 # Request timeout (seconds)
max_attempts = 5             # Retry attempts
cache_api_calls = True       # Enable caching to reduce costs
```

### Simulation Settings

```ini
[Simulation]
parallel_agent_actions = True    # Enable parallel execution
max_workers = 3                  # Thread pool size
max_turns = 6                    # Default conversation turns
max_personas = 10                # Maximum personas per simulation
enable_monitoring = True         # Enable system monitoring
enable_validation = True         # Enable persona validation
auto_analysis = True             # Automatic post-simulation analysis
```

### Monitoring & Alerts

```ini
[Monitoring]
log_level = INFO                 # DEBUG, INFO, WARNING, ERROR
performance_tracking = True      # Track response times
enable_metrics_export = False    # Enable Prometheus export
metrics_port = 9090              # Metrics server port
```

### Output Settings

```ini
[Output]
save_visualizations = True       # Auto-save charts
export_format = json             # Default export format
output_dir = outputs             # Output directory
```

---

## 🏗️ Architecture

### Core Components

#### 1. Simulation Engine (`scripts/simulation_engine.py`)
- **SimulationRunner**: Main orchestrator class
- **Agent Lifecycle Management**: Cleanup, reset, unique naming
- **Validation Integration**: Real-time persona validation
- **Parallel Execution**: ThreadPoolExecutor for concurrent actions
- **Memory Management**: Session clearing and registry cleanup

Key methods:
```python
run_simulation(config: SimulationConfig) -> SimulationResult
reset_agents_for_reuse(agents: List[TinyPerson])
clear_session_memory(agent: TinyPerson)
```

#### 2. Persona Validator (`scripts/persona_validator.py`)
- **ValidationResult**: Scores, warnings, suggestions
- **7 Validation Types**: Trait, role, realism, depth, tone, conversation, batch
- **Scoring System**: 0-100 scale for consistency and realism

Key methods:
```python
validate_persona(persona_dict: dict) -> ValidationResult
validate_conversation_consistency(agent: TinyPerson, context: str) -> ValidationResult
validate_batch(personas: List[dict]) -> List[ValidationResult]
```

#### 3. Security Manager (`scripts/security_manager.py`)
- **SecurityManager**: Authentication and access control
- **SessionManager**: Session lifecycle management
- **RateLimiter**: Request rate limiting
- **AuditLogger**: Security event logging

Key methods:
```python
authenticate(username: str, password: str) -> bool
check_rate_limit(user_id: str) -> bool
sanitize_input(text: str) -> str
```

#### 4. Backup Manager (`scripts/backup_manager.py`)
- **BackupManager**: Backup creation and restoration
- **BackupConfig**: Configuration management
- **Integrity verification**: Checksum validation

Key methods:
```python
create_backup(name: str = None) -> dict
restore_backup(backup_name: str, create_safety_backup: bool = True) -> dict
list_backups() -> List[dict]
```

#### 5. System Monitor (`scripts/monitoring.py`)
- **SystemMonitor**: Metrics collection and tracking
- **MonitoringConfig**: Threshold configuration
- **HealthStatus**: System health determination

Key methods:
```python
record_request(response_time: float, error: bool = False, error_type: str = None)
get_health_status() -> dict
export_prometheus_metrics() -> str
```

### Data Flow

```
User Input → Streamlit UI (app_updated.py)
    ↓
Persona Validation (persona_validator.py)
    ↓
Simulation Configuration (SimulationConfig)
    ↓
Simulation Execution (simulation_engine.py)
    ↓
TinyTroupe Framework (TinyPerson, TinyWorld)
    ↓
OpenAI API (GPT models)
    ↓
Results Processing (Analysis, Visualization)
    ↓
Display & Export (6 tabs: Conversation, Thoughts, Chat, Analysis, Metrics, Export)
```

### Session State Management

Streamlit session state stores:
- `personas_data`: All available personas
- `custom_personas`: User-created personas
- `conversation_history`: Full conversation log
- `last_simulation_agents`: Agent objects for reuse
- `agent_chat_history_{id}`: Per-agent chat history
- `last_analysis`: Analysis results
- `last_visualizations`: Generated charts

---

## 🛡️ Production Deployment

### Security Hardening

1. **Enable Authentication**
   ```python
   # In app_updated.py
   USE_AUTHENTICATION = True
   ```

2. **Configure Strong Credentials**
   ```ini
   # In config/config.ini or .env
   SECRET_KEY = <32+ character random string>
   SESSION_TIMEOUT = 3600
   MAX_LOGIN_ATTEMPTS = 3
   ```

3. **Enable HTTPS**
   ```bash
   streamlit run app_updated.py --server.port=443 --server.enableCORS=false --server.enableXsrfProtection=true
   ```

4. **Configure Rate Limits**
   ```python
   config = SecurityConfig(
       rate_limit_per_minute=100,
       enable_audit_logging=True
   )
   ```

### Backup Configuration

```python
from scripts.backup_manager import BackupManager, BackupConfig

config = BackupConfig(
    backup_dir="backups",
    auto_backup_enabled=True,
    backup_interval_minutes=60,
    max_backups=10,
    max_age_days=30
)

manager = BackupManager(config)
manager.start()  # Starts automatic backup thread
```

### Monitoring Setup

```python
from scripts.monitoring import SystemMonitor, MonitoringConfig

config = MonitoringConfig(
    enable_monitoring=True,
    health_check_interval=60,
    alert_thresholds={
        'cpu_percent': 80.0,
        'memory_percent': 85.0,
        'error_rate_per_minute': 10.0,
        'avg_response_time_seconds': 5.0
    }
)

monitor = SystemMonitor(config)
monitor.start()  # Starts monitoring thread

# Check health
health = monitor.get_health_status()
print(f"Status: {health['status']}")
```

### Prometheus Integration

```python
from scripts.monitoring import start_metrics_server

# Start metrics server on port 9090
start_metrics_server(monitor, port=9090)
```

Access metrics at: `http://localhost:9090/metrics`

Health check at: `http://localhost:9090/health`

### Load Balancing

Configure parallel execution in `config/config.ini`:

```ini
[Simulation]
parallel_agent_actions = True
max_workers = 3              # Adjust based on CPU cores
rate_limit_delay = 1.0       # Seconds between API calls
```

---

## 🔧 Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| **"OpenAI API key not found"** | Missing or incorrect API key | Add `api_key = sk-...` to `config/config.ini` under `[OpenAI]` |
| **"Agent name already in use"** | Old registry entries | Restart application (automatic registry cleanup implemented) |
| **"Rate limit exceeded"** | Too many API calls | Increase `rate_limit_delay` in config.ini or wait before retrying |
| **Slow simulation** | Sequential execution | Enable `parallel_agent_actions = True` in config.ini |
| **High memory usage** | Agent accumulation | Automatic cleanup after each simulation (no action needed) |
| **"Module not found"** | Missing dependencies | Run `pip install -r requirements.txt` |
| **Validation warnings** | Inconsistent persona | Review warnings and adjust persona characteristics |
| **Chat not working** | No previous simulation | Run a simulation first to create agents |
| **Backup failed** | Insufficient disk space | Free up disk space or reduce `max_backups` |
| **Health check unhealthy** | Resource exhaustion | Check alerts, restart application if needed |

### Debug Mode

Enable detailed logging:

```python
# In app_updated.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or in `config/config.ini`:
```ini
[Monitoring]
log_level = DEBUG
```

### API Issues

**Rate Limits**:
- Free tier: 3 requests/minute, 200/day
- Increase delay: `rate_limit_delay = 2.0`
- Enable caching: `cache_api_calls = True`

**Timeouts**:
- Increase timeout: `timeout = 120`
- Reduce response length: `max_tokens = 2048`

**Invalid API Key**:
1. Verify key format starts with `sk-`
2. Check key has not expired
3. Ensure billing is set up on OpenAI account

### Performance Optimization

1. **Enable Caching**
   ```ini
   cache_api_calls = True
   ```

2. **Parallel Execution**
   ```ini
   parallel_agent_actions = True
   max_workers = 3
   ```

3. **Reduce Response Size**
   ```ini
   max_tokens = 2048
   ```

4. **Use Faster Model**
   ```ini
   model = gpt-4o-mini
   ```

### Recovery Procedures

**Data Loss**:
```python
from scripts.backup_manager import BackupManager

manager = BackupManager.from_config("config/config.ini")
backups = manager.list_backups()
print(backups)  # Find appropriate backup
manager.restore_backup("backup_20251114_153000")
```

**Corrupted Data**:
1. Check logs: `logs/audit.log`, `logs/simulation.log`
2. Verify integrity: Automatic checksum validation
3. Restore from backup: See above

**System Unresponsive**:
1. Check health: `http://localhost:9090/health`
2. Review alerts in monitoring dashboard
3. Restart application: `Ctrl+C` then relaunch

---

## 📚 API Reference

### Simulation Configuration

```python
from scripts.simulation_engine import SimulationConfig

config = SimulationConfig(
    topic="AI Ethics",
    max_turns=5,
    personas=[
        {
            "name": "Dr. Sarah Chen",
            "role": "AI Ethicist",
            "characteristics": ["thoughtful", "analytical", "ethical"],
            "description": "Specializes in AI ethics and policy",
            "goals": "Ensure AI development aligns with human values",
            "communication_style": "Measured and thoughtful"
        }
    ],
    parallel_execution=True,
    rate_limit_delay=1.0
)
```

### Running Simulations

```python
from scripts.simulation_engine import SimulationRunner

runner = SimulationRunner()
result = runner.run_simulation(config)

print(f"Success: {result.success}")
print(f"Agents created: {len(result.agent_objects)}")
print(f"Conversation: {result.conversation}")
print(f"Thoughts: {result.ai_thoughts}")
```

### Validating Personas

```python
from scripts.persona_validator import PersonaValidator

validator = PersonaValidator()

persona = {
    "name": "John Doe",
    "role": "Software Engineer",
    "characteristics": ["innovative", "detail-oriented"],
    "description": "Full-stack developer with 10 years experience",
    "goals": "Build scalable systems",
    "communication_style": "Technical and precise"
}

result = validator.validate_persona(persona)

print(f"Consistency Score: {result.consistency_score}/100")
print(f"Realism Score: {result.realism_score}/100")
print(f"Warnings: {result.warnings}")
print(f"Suggestions: {result.suggestions}")
```

### Security Operations

```python
from scripts.security_manager import SecurityManager, SecurityConfig

config = SecurityConfig(
    require_authentication=True,
    session_timeout_seconds=3600,
    max_failed_attempts=3,
    lockout_duration_seconds=300
)

security = SecurityManager(config)

# Authenticate user
if security.authenticate("username", "password"):
    session_id = security.create_session("username")
    print(f"Session created: {session_id}")
```

### Backup Operations

```python
from scripts.backup_manager import BackupManager, BackupConfig

config = BackupConfig(
    backup_dir="backups",
    auto_backup_enabled=True,
    backup_interval_minutes=60
)

manager = BackupManager(config)

# Create backup
result = manager.create_backup("pre_deployment")
print(f"Backup created: {result['backup_name']}")

# List backups
backups = manager.list_backups()
for backup in backups:
    print(f"{backup['name']}: {backup['size_mb']} MB, {backup['created']}")

# Restore backup
result = manager.restore_backup("backup_20251114_153000")
print(f"Restored: {result['success']}")
```

### Monitoring Operations

```python
from scripts.monitoring import SystemMonitor, MonitoringConfig

config = MonitoringConfig(
    enable_monitoring=True,
    alert_thresholds={
        'cpu_percent': 80.0,
        'memory_percent': 85.0
    }
)

monitor = SystemMonitor(config)

# Record request
monitor.record_request(response_time=1.5, error=False)

# Get health
health = monitor.get_health_status()
print(f"Status: {health['status']}")
print(f"Alerts: {health['alerts']}")

# Get metrics
metrics = monitor.get_current_metrics()
print(f"CPU: {metrics['cpu_percent']}%")
print(f"Memory: {metrics['memory_percent']}%")
```

---

## 📁 Project Structure

```
P2 Deliverable 3/
├── app_updated.py                  # Main Streamlit application
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── launch_app.bat                  # Windows launcher
├── launch_app.ps1                  # PowerShell launcher
├── config/
│   └── config.ini                  # Configuration settings
├── data/
│   ├── personas.agents.json        # Pre-defined personas (50+)
│   └── feature.prompts.json        # Conversation topics (100+)
├── scripts/
│   ├── __init__.py
│   ├── simulation_engine.py        # Core simulation logic
│   ├── persona_validator.py        # Persona validation system
│   ├── security_manager.py         # Authentication & security
│   ├── backup_manager.py           # Backup & recovery
│   └── monitoring.py               # System monitoring
├── outputs/                        # Simulation results
│   ├── conversation_log.jsonl
│   ├── analysis_conversation_log.md
│   ├── conversation_readme.md
│   └── visualizations/             # Generated charts
├── backups/                        # Automatic backups
├── logs/                           # Application logs
│   ├── audit.log
│   ├── simulation.log
│   └── monitoring.log
└── __pycache__/                    # Python cache
```

---

## 🧪 Testing

### Running Unit Tests

```bash
# Test persona validation
python -m pytest tests/test_persona_validator.py

# Test simulation engine
python -m pytest tests/test_simulation_engine.py

# Test all
python -m pytest tests/
```

### Manual Testing Checklist

- [ ] Create custom persona with validation warnings
- [ ] Run simulation with AI-selected personas
- [ ] Run simulation with manually selected personas
- [ ] Test individual agent chat
- [ ] Test group agent chat
- [ ] View AI thoughts for each turn
- [ ] Check analysis tab for metrics
- [ ] Export conversation in all formats
- [ ] Create manual backup
- [ ] Restore from backup
- [ ] Check monitoring health status
- [ ] Test authentication (if enabled)

---

## 📊 Benchmarks & Performance

### API Usage

- **Average tokens per conversation**: ~2,000-5,000 (depends on turns and personas)
- **Cost per simulation** (gpt-4o-mini): $0.01-0.05
- **Cost per simulation** (gpt-4): $0.10-0.50

### Performance Metrics

| Configuration | Execution Time | Memory Usage |
|---------------|----------------|--------------|
| 3 agents, 3 turns, sequential | ~30-45 seconds | ~200 MB |
| 3 agents, 3 turns, parallel | ~15-25 seconds | ~250 MB |
| 5 agents, 5 turns, parallel | ~40-60 seconds | ~400 MB |

### Optimization Tips

1. **Enable parallel execution**: 40-50% faster
2. **Use caching**: Reduce duplicate API calls
3. **Smaller models**: gpt-4o-mini is 10x cheaper, similar quality
4. **Limit tokens**: Reduce `max_tokens` for shorter responses

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- Additional validation checks
- More conversation topics
- Enhanced visualization types
- Additional export formats
- Performance optimizations
- Testing coverage

---

## 📄 License

This project builds upon Microsoft TinyTroupe, licensed under MIT.

---

## 🙏 Acknowledgments

- **Microsoft TinyTroupe**: Foundation framework for multi-agent simulation
- **OpenAI**: GPT models powering agent intelligence
- **Streamlit**: Interactive web interface framework

---

## 📞 Support

For issues or questions:

1. Check troubleshooting section above
2. Review configuration settings
3. Check logs in `logs/` directory
4. Verify OpenAI API key and billing

---

## 🔄 Version History

**v3.0** (Current)
- Added persona validation system (7 validation types)
- Integrated validation into custom persona creator
- Real-time validation feedback in UI

**v2.0**
- Added production features (security, backup, monitoring)
- Implemented optional authentication
- Added automatic backups with rotation
- System monitoring with health checks
- Prometheus metrics export

**v1.0**
- Initial release with agent reuse
- Individual and group agent chat
- AI thoughts capture
- Comprehensive analysis and visualization
- Export capabilities (JSON, JSONL, Markdown)

---

**Last Updated**: January 2025
**Status**: Production Ready ✅
