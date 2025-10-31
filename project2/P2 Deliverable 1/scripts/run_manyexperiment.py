#!/usr/bin/env python3
"""
run_experiment.py — Run controlled experiments with TinyTroupe's InPlaceExperimentRunner.

Examples:
- Compare different prompts/seeds
- Run multiple trials with same settings
- A/B test different agent configurations
- Measure impact of temperature/model changes

Usage:
    python run_experiment.py --runs 3 --variants baseline improved
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

from tinytroupe.agent import TinyPerson
from tinytroupe.environment import TinyWorld
from tinytroupe.experimentation import InPlaceExperimentRunner
from tinytroupe import config_manager

# Reuse config and paths from main script
from run_simulation import (
    CFG, ROOT, DATA_DIR, OUT_DIR,
    MODEL, TEMPERATURE, MAX_TURNS,
    load_personas, _iter_agents
)

# --------------------
# Experiment variants
# --------------------

VARIANTS = {
    "baseline": {
        "seed": (
            "Debate whether to ship an auto-context feature that summarizes "
            "the last 3 minutes of a livestream for late-joining viewers."
        ),
        "temperature": 0.7,
        "model": MODEL,
    },
    "improved": {
        "seed": (
            "As a product team, evaluate the proposal: Auto-generate context summaries "
            "for late-joining livestream viewers. The system would analyze the last 3 "
            "minutes and display key points. Consider user value, technical feasibility, "
            "and potential risks."
        ),
        "temperature": 0.5,  # More focused responses
        "model": MODEL,
    }
}

# --------------------
# Experiment setup
# --------------------

class LivestreamDiscussion:
    """Experiment definition for the InPlaceExperimentRunner."""
    
    def __init__(self, variant_config: Dict[str, Any]):
        self.config = variant_config
        self.agents = load_personas(DATA_DIR / "personas.agent.json")
        
        # Apply variant-specific settings to all agents
        for agent in self.agents:
            agent.temperature = variant_config["temperature"]
            agent.model = variant_config["model"]
    
    def setup(self):
        """Create discussion room and make agents visible to each other."""
        self.world = TinyWorld(name="DiscussionRoom", agents=self.agents)
        try:
            self.world.make_everyone_accessible()
        except Exception:
            pass  # Optional in some versions
        
        # Initialize transcript for this run
        self.transcript = []
        
        # Broadcast the variant-specific seed
        self.world.broadcast(self.config["seed"])
    
    def step(self):
        """Run one full round of discussion (each agent speaks once)."""
        for agent in _iter_agents(self.world):
            msg = agent.listen_and_act(
                "Continue the group discussion with one concise, concrete point "
                "and an implicit handoff."
            )
            text = msg if isinstance(msg, str) else getattr(msg, "text", str(msg))
            self.transcript.append({
                "variant": self.config.get("name", "unknown"),
                "speaker": agent.name,
                "role": agent.role,
                "text": (text or "").strip(),
                "temperature": self.config["temperature"],
                "model": self.config["model"]
            })
    
    def run(self, turns: int):
        """Run the full experiment for specified turns."""
        self.setup()
        for _ in range(turns):
            self.step()
        return self.transcript

def save_results(results: List[Dict], run_id: str):
    """Save experiment results in both JSONL and readable formats."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = OUT_DIR / "experiments" / f"{ts}_{run_id}"
    base.parent.mkdir(parents=True, exist_ok=True)
    
    # Save raw results
    with (base / "results.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    # Generate readable summary
    lines = [
        f"# Experiment Results - {run_id} ({ts})",
        "",
        "## Configuration",
        "```json",
        json.dumps(VARIANTS, indent=2),
        "```",
        "",
        "## Transcripts by Variant",
        ""
    ]
    
    by_variant = {}
    for r in results:
        variant = r["variant"]
        if variant not in by_variant:
            by_variant[variant] = []
        by_variant[variant].append(r)
    
    for variant, entries in by_variant.items():
        lines.extend([
            f"### {variant}",
            f"- Temperature: {VARIANTS[variant]['temperature']}",
            f"- Model: {VARIANTS[variant]['model']}",
            f"- Seed: {VARIANTS[variant]['seed']}",
            ""
        ])
        for e in entries:
            lines.extend([
                f"**{e['speaker']} ({e['role']})**:",
                e['text'],
                ""
            ])
    
    (base / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return base

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1,
                      help="Number of times to run each variant")
    parser.add_argument("--turns", type=int, default=MAX_TURNS,
                      help="Discussion turns per run")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()),
                      help="Which variants to run (from VARIANTS dict)")
    parser.add_argument("--run-id", default="livestream_discussion",
                      help="Identifier for this experiment batch")
    args = parser.parse_args()
    
    # Always enable parallel for experiments
    config_manager.update("parallel", True)
    
    # Set up runner with variant configs
    runner = InPlaceExperimentRunner()
    all_results = []
    
    for variant_name in args.variants:
        if variant_name not in VARIANTS:
            print(f"Warning: Skipping unknown variant '{variant_name}'")
            continue
        
        config = VARIANTS[variant_name].copy()
        config["name"] = variant_name
        
        # Run multiple trials of this variant
        for run in range(args.runs):
            print(f"Running {variant_name} (trial {run + 1}/{args.runs})...")
            experiment = LivestreamDiscussion(config)
            results = experiment.run(args.turns)
            all_results.extend(results)
    
    # Save all results
    output_dir = save_results(all_results, args.run_id)
    print(f"Saved experiment results to: {output_dir}")

if __name__ == "__main__":
    main()