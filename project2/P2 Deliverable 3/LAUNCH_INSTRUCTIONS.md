# TinyTroupe Simulation Engine - Production Launch Guide

> **🚀 New Enhanced Version**: This guide covers launching the consolidated, production-ready TinyTroupe simulation engine with advanced features including real-time monitoring, enhanced persona validation, and comprehensive analytics.

## ⚡ Quick Start (Recommended)

### Option 1: One-Click Launch
**Windows Batch (Easiest)**:
1. Double-click `launch_app.bat` 
2. Browser opens automatically at http://localhost:8501
3. Enhanced UI with monitoring dashboard loads

**PowerShell (Advanced)**:
1. Right-click `launch_app.ps1` → "Run with PowerShell"
2. For first-time users, enable scripts:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

## 🖥️ Create Desktop Shortcuts

### Batch File Shortcut:
1. Right-click `launch_app.bat` → "Create shortcut"
2. Drag shortcut to Desktop
3. Optional: Right-click shortcut → Properties → Change Icon

### PowerShell Shortcut:
1. Right-click Desktop → New → Shortcut
2. Target location:
   ```
   powershell.exe -ExecutionPolicy Bypass -File "[PATH_TO_PROJECT]\launch_app.ps1"
   ```
3. Name: "TinyTroupe Production"

## 🎯 Alternative Launch Methods

### Command Line Interface
```bash
# Navigate to project directory
cd "C:\Users\jakem\Documents\GitHub\cs676\project2\P2 Deliverable 2"

# Launch web interface
streamlit run app_updated.py

# Or use command line engine directly
python scripts/simulation_engine.py --help
```

### Direct Engine Usage (No UI)
```bash
# Single simulation
python scripts/simulation_engine.py --mode single --seed "Your topic here" --turns 6

# Batch experiments
python scripts/simulation_engine.py --mode experiment --variants baseline creative --runs 3

# Validate persona database
python scripts/simulation_engine.py --mode validate
```

## 🔧 Configuration Options

### API Key Setup (Choose one):

**1. Environment Variable (Recommended)**:
```bash
set OPENAI_API_KEY=your-api-key-here
```

**2. Streamlit Secrets**:
Create `.streamlit/secrets.toml`:
```toml
[openai]
api_key = "your-api-key-here"
```

**3. Config File**:
Edit `config/config.ini`:
```ini
[openai]
api_key = your-api-key-here
```

### Advanced Configuration
Edit `config/config.ini` for production settings:
```ini
[OpenAI]
model = gpt-4o-mini
temperature = 0.7

[Simulation]
parallel_agent_actions = True
max_turns = 6

[Performance]
max_workers = 3
rate_limit_delay = 1.0
```

## ✨ New Features Available

### Enhanced Web Interface (`app_updated.py`)
- 📊 **Real-time Monitoring**: Performance metrics during execution
- 🎛️ **Advanced Controls**: Parallel execution, rate limiting, cost tracking
- 👥 **Dynamic Personas**: Create and validate personas through UI
- 📈 **Live Analytics**: Automatic conversation analysis with visualizations
- 🛡️ **Security Features**: Session limits, API usage tracking
- 💾 **Multiple Export Formats**: JSON, JSONL, Markdown with analysis

### Consolidated Engine (`scripts/simulation_engine.py`)
- 🎯 **Unified API**: Single script replaces 4 separate tools
- 🚀 **Production Ready**: Error handling, monitoring, load balancing
- 🔍 **Comprehensive Validation**: 15+ persona validation rules
- ⚡ **Performance Optimized**: Parallel execution, smart batching
- 📊 **Built-in Analytics**: Automatic conversation analysis

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| "streamlit not found" | Run `pip install -r requirements.txt` |
| "Import errors" | Check virtual environment activation |
| "API key missing" | Set OPENAI_API_KEY environment variable |
| "Port already in use" | Use `streamlit run app_updated.py --server.port 8502` |
| "Permission denied" | Run PowerShell as administrator once to set execution policy |
| "Persona validation errors" | Check `data/personas.agents.json` format |

## 📊 System Requirements

- **Python**: 3.8+ 
- **Memory**: 4GB+ recommended for large simulations
- **Storage**: 1GB+ for outputs and cache
- **Network**: Stable connection for OpenAI API

## 🎮 Usage Tips

1. **Start Small**: Begin with 3-4 personas and 4-6 turns
2. **Monitor Performance**: Use the built-in metrics dashboard
3. **Enable Caching**: Speeds up repeated runs significantly
4. **Use Parallel Mode**: For better performance with multiple personas
5. **Check Validation**: Always validate personas before large runs

## 📞 Getting Help

- **Documentation**: See `DOCUMENTATION.md` for complete guide
- **Debug Info**: Enable "Show Debug Info" in the web interface
- **Logs**: Check terminal output for detailed error messages
- **Config Issues**: Verify `config/config.ini` settings

---

**Ready to start?** Double-click `launch_app.bat` and explore the enhanced interface! 🚀
