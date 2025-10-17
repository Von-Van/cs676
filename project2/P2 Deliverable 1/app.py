#!/usr/bin/env python3
"""
Streamlit app — Persona-based Feedback Simulator (TinyTroupe)

Features:
- Input feature text
- Choose predefined personas or add a custom persona
- Run simulation (calls simulate.py) and store last run path
- View evaluation report (calls evaluate.py)
- Export conversation history (calls export_conversation_history.py)
- Display transcripts per persona in tabs with assistant metadata

Run:
  streamlit run app.py
"""

import io
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import streamlit as st

PROJECT_DIR = Path(__file__).parent
RUNS_DIR = PROJECT_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Persona Feedback Simulator", layout="wide")
st.title("Persona-based Feedback Simulator (TinyTroupe)")

# -------- Personas load + custom creation --------

with open(PROJECT_DIR / "personas.json", "r", encoding="utf-8") as f:
    personas = json.load(f)

persona_ids = [p["id"] for p in personas]
persona_by_id = {p["id"]: p for p in personas}

st.sidebar.header("Personas")
selected_ids = st.sidebar.multiselect("Choose personas", persona_ids, default=persona_ids[:2])

st.sidebar.markdown("**Add a custom persona**")
with st.sidebar.expander("Custom persona"):
    custom_name = st.text_input("Name", "")
    custom_traits = st.text_input("Traits (comma-separated)", "")
    custom_goals = st.text_input("Goals (comma-separated)", "")
    custom_demo = st.text_input("Demographics (free text)", "")
    custom_style = st.text_area("Style notes", "")

def build_custom():
    if not custom_name:
        return None
    return {
        "id": f"custom_{int(time.time())}",
        "name": custom_name,
        "demographics": {"notes": custom_demo} if custom_demo else {},
        "traits": [t.strip() for t in custom_traits.split(",") if t.strip()],
        "goals": [g.strip() for g in custom_goals.split(",") if g.strip()],
        "style": custom_style or "—",
    }

custom_persona = build_custom()
if custom_persona:
    personas.append(custom_persona)
    persona_by_id[custom_persona["id"]] = custom_persona
    if custom_persona["id"] not in selected_ids:
        selected_ids.append(custom_persona["id"])

# -------- Feature & controls --------

feature_text = st.text_area(
    "Feature description",
    placeholder="Describe the feature, flows, visuals, and context...",
    height=160,
)

turns = st.slider("Turns per persona", 2, 8, 4)

colA, colB, colC = st.columns([1,1,1])
with colA:
    run_btn = st.button("Run Simulation")
with colB:
    eval_btn = st.button("Evaluate Last Run")
with colC:
    export_btn = st.button("Export Transcript (.md)")

# -------- Helpers to invoke scripts --------

def write_temp_personas(ids):
    selected = [persona_by_id[i] for i in ids if i in persona_by_id]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    tmp.write(json.dumps(selected, ensure_ascii=False).encode("utf-8"))
    tmp.close()
    return tmp.name

def run_sim(turns, feature, ids):
    if not ids:
        return None, "", "No personas selected."
    pfile = write_temp_personas(ids)
    out = RUNS_DIR / f"{int(time.time())}_run.jsonl"
    cmd = ["python", "simulate.py", "--turns", str(turns), "--personas", pfile, "--out", str(out)]
    env = os.environ.copy()
    env["FEATURE_TEXT"] = feature or ""
    res = subprocess.run(cmd, cwd=str(PROJECT_DIR), capture_output=True, text=True, env=env)
    return str(out), res.stdout, res.stderr

def run_eval(path):
    rep = RUNS_DIR / (Path(path).stem + "_report.md")
    cmd = ["python", "evaluate.py", "--in", path, "--report", str(rep), "--personas", "personas.json"]
    res = subprocess.run(cmd, cwd=str(PROJECT_DIR), capture_output=True, text=True)
    return str(rep), res.stdout, res.stderr

def run_export(path):
    md = PROJECT_DIR / "conversation_history.md"
    cmd = ["python", "export_conversation_history.py", "--in", path, "--out", str(md), "--personas", "personas.json"]
    res = subprocess.run(cmd, cwd=str(PROJECT_DIR), capture_output=True, text=True)
    return str(md), res.stdout, res.stderr

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

# -------- Actions --------

if run_btn:
    if not feature_text:
        st.warning("Please enter a feature description.")
    elif not selected_ids:
        st.warning("Please choose at least one persona.")
    else:
        run_path, out, err = run_sim(turns, feature_text, selected_ids)
        if err.strip():
            st.error(err)
        if run_path and os.path.exists(run_path):
            st.session_state["last_run_path"] = run_path
            st.success(f"Saved transcripts to: {run_path}")
            # Show quick preview as tabs with chat bubbles
            rows = load_jsonl(run_path)
            by_pid = {}
            for r in rows:
                pid = r.get("persona_id", "unknown")
                by_pid.setdefault(pid, []).append(r)
            tabs = st.tabs([persona_by_id.get(pid, {"name": pid}).get("name", pid) for pid in by_pid.keys()])
            for tab, (pid, turns_list) in zip(tabs, by_pid.items()):
                with tab:
                    # sort by turn index
                    turns_list = sorted(turns_list, key=lambda x: x.get("turn_index", 0))
                    feat = next((t.get("feature_text") for t in turns_list if t.get("feature_text")), "")
                    if feat:
                        st.markdown(f"**Feature under review:** {feat}")
                    for t in turns_list:
                        role = t.get("role", "").capitalize()
                        content = (t.get("content") or "").strip()
                        if role == "User":
                            st.chat_message("user").markdown(content)
                        else:
                            with st.chat_message("assistant"):
                                st.markdown(content)
                                rs = t.get("reasoning_summary")
                                conf = t.get("confidence")
                                fus = t.get("followups") or []
                                meta_bits = []
                                if rs: meta_bits.append(f"**Reasoning:** {rs}")
                                if conf is not None: meta_bits.append(f"**Confidence:** {conf:.2f}")
                                if fus: meta_bits.append("**Follow-ups:** " + "; ".join(str(x) for x in fus))
                                if meta_bits:
                                    st.markdown("> " + "  \n> ".join(meta_bits))

if eval_btn:
    run_path = st.session_state.get("last_run_path")
    if not run_path or not os.path.exists(run_path):
        st.warning("No recent run found. Run a simulation first.")
    else:
        rep_path, out, err = run_eval(run_path)
        if err.strip():
            st.error(err)
        if os.path.exists(rep_path):
            st.success(f"Report written to: {rep_path}")
            with open(rep_path, "r", encoding="utf-8") as f:
                st.markdown(f.read())

if export_btn:
    run_path = st.session_state.get("last_run_path")
    if not run_path or not os.path.exists(run_path):
        st.warning("No recent run found. Run a simulation first.")
    else:
        md_path, out, err = run_export(run_path)
        if err.strip():
            st.error(err)
        if os.path.exists(md_path):
            st.success(f"Transcript exported to: {md_path}")
            with open(md_path, "r", encoding="utf-8") as f:
                st.download_button("Download conversation_history.md", f.read(), file_name="conversation_history.md")
