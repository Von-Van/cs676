# AI Agent Conversation Simulator

Simulated conversations between AI personas discussing product features and technical decisions. Built with TinyTroupe for persona-based feedback simulation.

## Project Structure

```
project2/
├── config/
│   └── config.ini       # Runtime configuration
├── data/
│   └── personas.json    # Agent specifications
├── outputs/             # Simulation results
│   ├── conversation_log.jsonl
│   └── conversation_readme.md
├── scripts/
│   ├── run_simulation.py      # Single-run simulator
│   ├── run_experiment.py      # Multi-variant experiments
│   └── analyze_conversations.py # Analysis tools
└── README.md
```

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure settings in `config/config.ini`
   ```ini
   [OpenAI]
   model = gpt-4.1-mini
   temperature = 0.4
   ```

3. Run a simulation:
   ```bash
   python scripts/run_simulation.py
   ```

## Agent Personas

- **Lisa Park (Data Scientist)**: Metric-driven experimentalist focused on measurable impact
- **Kai Ortega (Product Manager)**: Outcome-oriented PM aligning roadmap with user value
- **Rowan Shah (ML Engineer)**: Reliability-focused engineer emphasizing robust systems
- **Mira Iyengar (UX Researcher)**: Evidence-first researcher blending qual and quant methods
- **Eli Navarro (AI Ethics)**: Governance-minded ethicist ensuring transparency and safety

## Features

- 🤖 Persona-based conversations using TinyTroupe
- 📊 Multi-variant experiment support
- 📈 Detailed conversation analysis
- 📝 Rich conversation transcripts
- 🔄 Parallel execution support

## Example Conversation

The agents discuss product features through their unique perspectives:

```markdown
Turn 1 — Lisa Park (Data Scientist):
Before committing to this auto-context feature, we need clear success metrics. 
I propose tracking: time-to-first-engagement for late joiners, context comprehension 
scores, and potential negative impact on stream performance.

Turn 1 — Rowan Shah (ML Engineer):
Agreed on metrics. From a systems perspective, we need to consider: 1) latency impact 
of real-time summarization, 2) accuracy degradation patterns, and 3) fallback behavior 
when summarization fails. What's our target p99 latency for summary generation?
```

## Running Experiments

Test different configurations:

```bash
python scripts/run_experiment.py --runs 3 --variants baseline improved
```

Analyze results:

```bash
python scripts/analyze_conversations.py --experiment outputs/experiments/results.jsonl
```

## Analysis Features

- Participation metrics
- Interaction patterns
- Content focus tracking
- Topic flow visualization
- Cross-variant comparisons

## Latest Results

Analysis from 2025-10-31 17:40:

- **Messages:** 12
- **Total Words:** 806
- **Average Length:** 67.2 words

**Top Interaction Patterns:**
```
sa_chen->mr_rivera: 2 exchanges
mr_rivera->ap_patel: 2 exchanges
ap_patel->jo_neil: 2 exchanges
```

**Content Focus:**
```
Technical: 6 mentions
UX/Design: 17 mentions
Data/Analytics: 4 mentions
```

See [conversation analysis](outputs\analysis_conversation_log.md) for details.

## Contributing

1. Clone the repository
2. Create a feature branch
3. Add/update tests
4. Submit a pull request
