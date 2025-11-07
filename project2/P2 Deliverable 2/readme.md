# AI Agent Conversation Simulator

An advanced persona-based feedback simulation system that leverages AI agents to generate realistic conversations about product features, technical decisions, and strategic initiatives. Built with TinyTroupe for sophisticated multi-agent simulations that provide early-stage validation and diverse perspective synthesis.

## Simulation Algorithm Design

### Core Architecture

The simulation system employs a sophisticated multi-agent architecture built on TinyTroupe, designed to model realistic human perspectives and decision-making processes in product development contexts.

#### 1. Persona Modeling Framework

**Cognitive Architecture**: Each AI persona is modeled as a `TinyPerson` with layered psychological and professional attributes:

- **Identity Layer**: Core biographical data (name, role, background) establishing professional context
- **Personality Layer**: Behavioral traits, decision-making patterns, and communication styles
- **Goal Layer**: Professional objectives, priorities, and success metrics that drive responses
- **Contextual Layer**: Domain expertise, experience level, and perspective framing

**Persona Specification Schema**:
```json
{
  "personality": {
    "traits": ["methodical", "forward-thinking", "detail-oriented"],
    "goals": ["Ensure system scalability", "Design elegant solutions"],
    "style": "Analytical and thorough",
    "communication": "structured",
    "decisions": "data-driven",
    "approach": "systematic"
  }
}
```

**Psychological Consistency**: The system maintains persona consistency through:
- **Memory Architecture**: Each agent maintains episodic and semantic memory of the conversation
- **Belief Systems**: Core professional values that influence all responses
- **Cognitive Biases**: Realistic decision-making patterns based on role and experience
- **Emotional State**: Contextual emotional responses that affect communication style

#### 2. Feature Description Processing

**Multi-Modal Input Processing**: The system accepts various feature description formats:

- **Natural Language Prompts**: Conversational feature descriptions for ideation phases
- **Technical Specifications**: Detailed requirements for engineering evaluation
- **User Stories**: Product requirements for user-centered analysis
- **Business Cases**: Strategic initiatives for market validation

**Semantic Analysis Pipeline**:
1. **Context Extraction**: Identifies key product domains, technical requirements, and business objectives
2. **Stakeholder Mapping**: Determines which personas are most relevant for the discussion
3. **Complexity Assessment**: Evaluates technical and business complexity to guide conversation depth
4. **Perspective Priming**: Prepares each agent with role-specific context and priorities

#### 3. Conversation Generation Engine

**Turn-Based Interaction Model**:
- **Structured Discourse**: Each turn represents ~15 minutes of real-time discussion
- **Dynamic Speaking Order**: Randomized agent order prevents artificial conversation patterns
- **Contextual Handoffs**: Agents naturally reference and build upon previous contributions
- **Parallel Processing**: Agents can process information simultaneously while maintaining conversational flow

**Response Generation Process**:
1. **Context Integration**: Agent processes full conversation history and current topic focus
2. **Role-Specific Analysis**: Applies professional lens to evaluate the feature/decision
3. **Perspective Synthesis**: Generates response based on personality, goals, and expertise
4. **Quality Validation**: Ensures responses are constructive, relevant, and in-character

**Conversation Control Mechanisms**:
- **Topic Management**: Prevents conversational drift through goal-oriented prompting
- **Participation Balance**: Ensures all relevant perspectives are represented
- **Depth Control**: Manages technical vs. strategic discussion levels
- **Conflict Resolution**: Handles disagreements constructively to reach synthesis

#### 4. Feedback Synthesis Algorithm

**Multi-Perspective Analysis Engine**:
- **Sentiment Analysis**: Tracks positive, negative, and neutral reactions across personas
- **Concern Categorization**: Groups feedback into technical, user experience, business, and ethical dimensions
- **Priority Scoring**: Weights feedback based on agent expertise and organizational impact
- **Consensus Detection**: Identifies areas of agreement and persistent disagreements

**Synthesis Process**:
1. **Content Analysis**: Extracts key insights, concerns, and recommendations from each turn
2. **Cross-Reference Mapping**: Identifies supporting and conflicting viewpoints
3. **Impact Assessment**: Evaluates potential business and technical implications
4. **Recommendation Generation**: Synthesizes actionable next steps and decision points

### AI Decision-Making Architecture

**Large Language Model Integration**:
- **Model Selection**: Configurable backend (GPT-4, GPT-3.5) based on complexity and cost requirements
- **Temperature Control**: Adjustable creativity vs. consistency based on simulation goals
- **Context Window Management**: Efficient conversation history management for long discussions
- **API Optimization**: Intelligent caching and rate limiting for cost-effective operation

**Prompt Engineering Strategy**:
- **Role-Specific Prompting**: Tailored system prompts for each professional persona
- **Context-Aware Instructions**: Dynamic prompting based on conversation state and objectives
- **Consistency Enforcement**: Mechanisms to maintain character and professional authenticity
- **Quality Control**: Built-in validation to ensure constructive, relevant contributions

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

> Tip: `analyze_conversations.py` now defaults to the most-recent conversation log in `outputs/` when no input file is supplied.

Analyze a specific file:

```cmd
python scripts/analyze_conversations.py --experiment outputs/experiments/results.jsonl
```

Analyze the most recent conversation log (no arguments):

```cmd
python scripts/analyze_conversations.py
```

Options:

- `--output-dir <path>` — write visualizations and report into a specific directory (defaults to the input file's parent).
- `--skip-readme` — skip updating the project `README.md` with a short summary of the analysis.

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
