import json
from pathlib import Path
from datetime import datetime
import configparser

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
CFG_PATH = ROOT / "config" / "config.ini"

LOG_PATH = OUT_DIR / "conversation_log.jsonl"
README_PATH = OUT_DIR / "conversation_readme.md"

CFG = configparser.ConfigParser()
CFG.read(CFG_PATH)
MODEL = CFG.get("model", "name", fallback="gpt-4.1-mini")
MAX_TURNS = CFG.getint("run", "max_turns", fallback=6)
SEED = CFG.get("run", "seed", fallback="<seed unknown>")

def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def guess_participants(rows):
    # preserve appearance order
    seen = {}
    for r in rows:
        key = (r.get("speaker",""), r.get("role",""))
        if key not in seen:
            seen[key] = True
    return list(seen.keys())

def write_markdown(rows, path: Path):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    parts = guess_participants(rows)
    lines = [
        f"# TinyTroupe Conversation — {ts}",
        "",
        f"- **Model:** `{MODEL}`",
        f"- **Turns:** {MAX_TURNS}",
        f"- **Seed:** {SEED}",
        "",
        "## Participants",
    ]
    for speaker, role in parts:
        lines.append(f"- **{speaker}** — {role}")
    lines += ["", "## Transcript", ""]

    for r in rows:
        lines.append(f"**Turn {r.get('turn','?')} — {r.get('speaker','?')} ({r.get('role','?')})**")
        lines.append((r.get("text","")).strip())
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")

if __name__ == "__main__":
    rows = read_jsonl(LOG_PATH)
    write_markdown(rows, README_PATH)
    print(f"Rewrote {README_PATH} from {LOG_PATH}")
