## 1️⃣ Environment Setup

### Requirements
- **Python 3.10 or later**  
- **VS Code** or terminal access  
- **Internet connection** (for package install and OpenAI API calls)

### Create and Activate a Virtual Environment
```cmd
cd "C:\Users\jakem\Documents\GitHub\cs676\project2\P2 Deliverable 1"
python -m venv ..\.venv
..\.\.venv\Scripts\activate.bat
```

Verify the environment:
```cmd
where python
pip --version
```
Prompt should show `(.venv)` at the beginning.

---

## 2️⃣ Install Dependencies

Install required packages:
```cmd
pip install -r requirements.txt
```

Uninstall the PyPI placeholder and install the real TinyTroupe:
```cmd
pip uninstall tinytroupe -y
pip install git+https://github.com/microsoft/tinytroupe.git
```

---

## 3️⃣ Configure OpenAI Backend (API Key)

TinyTroupe requires an LLM backend (OpenAI or Azure OpenAI).  
This project uses **OpenAI** for simplicity and compatibility.

### Obtain an API Key
1. Visit [https://platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys)  
2. Click **Create new secret key**  
3. Copy the key (`sk-...`)

### Set the Key for Your Session
```cmd
set OPENAI_API_KEY=sk-your-real-key-here
```

### Confirm the Key Loaded
```cmd
echo %OPENAI_API_KEY%
```
If you see your key, it’s active.

> 💡 To save it permanently:
> ```cmd
> setx OPENAI_API_KEY "sk-your-real-key-here"
> ```
> Then **close and reopen CMD** before continuing.

---

## 4️⃣ TinyTroupe Configuration (`config.ini`)

Create a file named `config.ini` in the same folder as `simulate.py` and `app.py`:

```ini
[OpenAI]
api_type = openai
model = gpt-4o-mini
temperature = 0.7
timeout = 120
max_tokens = 2048
```

> ✅ At runtime, TinyTroupe will print “Found custom config on: …\config.ini” if it was successfully loaded.

---

## 5️⃣ Verify TinyTroupe Connection

Create a temporary file `tt_smoke.py` for a sanity test:

```python
from tinytroupe.examples import create_lisa_the_data_scientist

print("Building TinyTroupe person...")
p = create_lisa_the_data_scientist()
p.listen("[SYSTEM] You are evaluating a feature.")
out = p.listen_and_act("Feature: a one-click order-tracking widget. Please react briefly.")
print("RESPONSE:", out)
```

Run:
```cmd
python tt_smoke.py
```

✅ Expected: a short response from “Lisa Carter,” indicating TinyTroupe is connected to OpenAI.  
❌ If you see an error about `OPENAI_API_KEY`, rerun the `set` command above in this CMD window.

---

## 6️⃣ Run the Simulation Pipeline

```cmd
set FEATURE_TEXT=Order-tracking widget on receipts page
python simulate.py --turns 3 --personas personas.json --out runs\main_run.jsonl
```

---

## 7️⃣ Evaluate and Export Results

Generate a metrics report:
```cmd
python evaluate.py --in runs\main_run.jsonl --report runs\main_report.md --personas personas.json
```

Export conversation history for annotation:
```cmd
python export_conversation_history.py --in runs\main_run.jsonl --out conversation_history.md --personas personas.json
```

Expected outputs:
- `runs\main_run.jsonl`  → raw conversation data  
- `runs\main_report.md`  → quality metrics (Realism, Consistency, Diversity, Confidence)  
- `conversation_history.md`  → annotated transcript with reasoning and follow-ups  

---

## 8️⃣ Launch the Interactive App (optional)

```cmd
streamlit run app.py
```

In the browser interface:
1. Enter a feature description  
2. Choose personas (or add a custom one)  
3. Click **Run Simulation** to see chat output  
4. Click **Evaluate Last Run** for metrics  
5. Click **Export Transcript** to download `.md` output  

---

## 9️⃣ Troubleshooting Guide

| Issue | Likely Cause | Fix |
|:--|:--|:--|
| `openai.OpenAIError: api_key must be set` | Environment variable not set in this session | Run `set OPENAI_API_KEY=...` again in the same CMD window |
| `ModuleNotFoundError: tinytroupe` | Placeholder PyPI version installed | `pip uninstall tinytroupe -y` then `pip install git+https://github.com/microsoft/tinytroupe.git` |
| `Config not found` | `config.ini` missing or in wrong folder | Place it in the same folder as `simulate.py` |
| Empty responses from TinyTroupe | Expired key or network error | Re-create OpenAI key and verify internet access |
| “Permission denied” on activation | Execution policy blocking PowerShell scripts | Run PowerShell as Administrator → `Set-ExecutionPolicy RemoteSigned` |

---

## 🔍 Verification Checklist

- [x] `config.ini` detected in TinyTroupe startup logs  
- [x] `OPENAI_API_KEY` visible (`echo %OPENAI_API_KEY%`)  
- [x] `simulate.py` produces `runs\main_run.jsonl`  
- [x] `evaluate.py` generates `runs\main_report.md`  
- [x] `export_conversation_history.py` creates `conversation_history.md`  
- [x] Optional: Streamlit app launches successfully on http://localhost:8501  

## 💻 System Requirements
| Component | Recommended |
|:--|:--|
| OS | Windows 10/11 (64-bit) or macOS 13+ |
| Python | 3.10 or later |
| Memory | 8 GB RAM minimum |
| Internet | Stable connection to OpenAI API |

### ⚙️ Performance Notes
Each simulation call uses a GPT-4o-mini model request per persona turn.  
Approximate run time = 5–10 s × (number of turns × personas).  
For best performance:  
- Keep `max_tokens ≤ 1024` in `config.ini`  
- Disable parallel modes on Windows (`parallel_agent_actions=False`)  
- Use smaller `--turns` for quick tests  
