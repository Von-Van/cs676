# Setup Walkthrough

## Environment
- Python 3.10+, venv in `./.venv`
- `pip install -r requirements.txt`

## Installation Steps
1. Create venv (PowerShell or cmd).
2. `pip install tinytroupe openai tiktoken pydantic==2.* python-dotenv rich`
3. `.env` with `OPENAI_API_KEY=...`

## Usage
```bash
python simulate.py --turns 3 --personas personas.json --out runs/test.jsonl
python evaluate.py --in runs/test.jsonl --report runs/test_report.md --personas personas.json
python export_conversation_history.py --in runs/test.jsonl --out conversation_history.md --personas personas.json
