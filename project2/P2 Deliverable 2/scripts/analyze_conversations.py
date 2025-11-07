#!/usr/bin/env python3
"""
analyze_conversations.py — Analyze TinyTroupe conversation transcripts.

Features:
- Speaker participation metrics
- Response pattern analysis
- Topic tracking and flow
- Sentiment and interaction dynamics
- Conversation structure visualization

Usage:
    python analyze_conversations.py path/to/conversation.jsonl
    python analyze_conversations.py --experiment path/to/experiment/results.jsonl
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

# Optional viz support
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False

# --------------------
# Analysis Functions
# --------------------

def load_transcript(path: Path) -> pd.DataFrame:
    """Load transcript into a DataFrame with parsed metadata."""
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    
    df = pd.DataFrame(rows)
    
    # Ensure text column exists and normalize it
    if 'text' not in df.columns:
        df['text'] = ''
    else:
        df['text'] = df['text'].fillna('').astype(str)

    # Derived columns
    df['word_count'] = df['text'].str.split().str.len()
    df['char_count'] = df['text'].str.len()

    # Normalize speaker/role/turn columns if missing
    if 'speaker' not in df.columns:
        # try common alternative keys
        if 'author' in df.columns:
            df['speaker'] = df['author']
        else:
            df['speaker'] = ''

    if 'role' not in df.columns:
        df['role'] = ''

    if 'turn' not in df.columns:
        # default to 1 if no turn information available
        df['turn'] = df.get('turn', 1)

    # Parse timestamp if present
    if 'timestamp' in df.columns:
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        except Exception:
            df['timestamp'] = None

    return df

def analyze_participation(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze speaker participation patterns."""
    total_messages = len(df)
    total_words = int(df['word_count'].sum()) if 'word_count' in df.columns else 0

    messages_per_turn = 0
    try:
        if 'turn' in df.columns and total_messages > 0:
            messages_per_turn = float(df.groupby('turn').size().mean())
    except Exception:
        messages_per_turn = 0

    by_speaker_counts = df['speaker'].value_counts().to_dict() if 'speaker' in df.columns else {}
    by_speaker_word = df.groupby('speaker')['word_count'].sum().to_dict() if 'speaker' in df.columns else {}
    by_speaker_avg = df.groupby('speaker')['word_count'].mean().to_dict() if 'speaker' in df.columns else {}

    by_role_counts = df['role'].value_counts().to_dict() if 'role' in df.columns else {}
    by_role_word = df.groupby('role')['word_count'].sum().to_dict() if 'role' in df.columns else {}
    by_role_avg = df.groupby('role')['word_count'].mean().to_dict() if 'role' in df.columns else {}

    results = {
        'total_messages': total_messages,
        'total_words': total_words,
        'messages_per_turn': messages_per_turn,

        'by_speaker': {
            'message_counts': by_speaker_counts,
            'word_counts': by_speaker_word,
            'avg_length': {k: float(v) for k, v in (by_speaker_avg or {}).items()}
        },

        'by_role': {
            'message_counts': by_role_counts,
            'word_counts': by_role_word,
            'avg_length': {k: float(v) for k, v in (by_role_avg or {}).items()}
        }
    }

    # Add timing between messages if available
    if 'timestamp' in df.columns and df['timestamp'].notna().any():
        try:
            df['time_delta'] = df['timestamp'].diff()
            results['timing'] = {
                'avg_delay': df['time_delta'].mean(),
                'max_delay': df['time_delta'].max(),
                'by_speaker_delay': df.groupby('speaker')['time_delta'].mean().to_dict() if 'speaker' in df.columns else {}
            }
        except Exception:
            results['timing'] = {}

    return results

def analyze_interaction_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze how agents interact with each other."""
    # Track who responds to whom
    response_matrix = defaultdict(lambda: defaultdict(int))
    if len(df) < 2 or 'speaker' not in df.columns:
        return {'response_patterns': {}, 'turn_taking': {'unique_patterns': 0, 'most_common_pairs': []}}

    for i in range(1, len(df)):
        prev = df.iloc[i-1]
        curr = df.iloc[i]
        response_matrix[prev['speaker']][curr['speaker']] += 1
    
    # Convert to regular dict for JSON
    response_counts = {k: dict(v) for k, v in response_matrix.items()}
    
    return {
        'response_patterns': response_counts,
        'turn_taking': {
            'unique_patterns': len(response_counts),
            'most_common_pairs': sorted(
                [
                    (f"{s1}->{s2}", count)
                    for s1, responses in response_counts.items()
                    for s2, count in responses.items()
                ],
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }
    }

def analyze_content(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze message content patterns."""
    # Simple keyword tracking
    tech_keywords = {'architecture', 'system', 'api', 'latency', 'monitoring'}
    ux_keywords = {'user', 'experience', 'interface', 'design', 'usability'}
    data_keywords = {'metrics', 'measurement', 'analysis', 'data', 'experiment'}
    
    def count_keywords(text: str, keywords: set) -> int:
        return sum(1 for word in text.lower().split() if word in keywords)
    if 'text' not in df.columns or df.empty:
        content_focus = {'technical': 0, 'ux': 0, 'data': 0}
        by_speaker = {}
    else:
        content_focus = {
            'technical': int(df['text'].apply(lambda x: count_keywords(x, tech_keywords)).sum()),
            'ux': int(df['text'].apply(lambda x: count_keywords(x, ux_keywords)).sum()),
            'data': int(df['text'].apply(lambda x: count_keywords(x, data_keywords)).sum())
        }

        by_speaker = {}
        if 'speaker' in df.columns:
            for speaker, speaker_df in df.groupby('speaker'):
                by_speaker[speaker] = {
                    'technical': int(speaker_df['text'].apply(lambda x: count_keywords(x, tech_keywords)).sum()) if 'text' in speaker_df.columns else 0,
                    'ux': int(speaker_df['text'].apply(lambda x: count_keywords(x, ux_keywords)).sum()) if 'text' in speaker_df.columns else 0,
                    'data': int(speaker_df['text'].apply(lambda x: count_keywords(x, data_keywords)).sum()) if 'text' in speaker_df.columns else 0
                }

    return {
        'content_focus': content_focus,
        'by_speaker': by_speaker
    }

def visualize_conversation(df: pd.DataFrame, output_dir: Path) -> List[Path]:
    """Generate visualization plots if matplotlib is available."""
    if not HAS_VIZ:
        return []
    
    generated_files = []
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Message distribution
    plt.figure(figsize=(10, 6))
    speaker_counts = df['speaker'].value_counts()
    sns.barplot(x=speaker_counts.index, y=speaker_counts.values)
    plt.title('Messages per Speaker')
    plt.xticks(rotation=45)
    path = output_dir / 'messages_per_speaker.png'
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    generated_files.append(path)
    
    # 2. Interaction heatmap
    plt.figure(figsize=(10, 8))
    pivot = pd.crosstab(df['speaker'], df['speaker'].shift())
    sns.heatmap(pivot, annot=True, fmt='d', cmap='YlGnBu')
    plt.title('Speaker Interaction Heatmap')
    path = output_dir / 'interaction_heatmap.png'
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    generated_files.append(path)
    
    # 3. Word count trends
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='turn', y='word_count', hue='speaker')
    plt.title('Message Length Over Time')
    path = output_dir / 'message_length_trends.png'
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    generated_files.append(path)
    
    return generated_files

def write_analysis_report(analysis: Dict[str, Any], 
                         df: pd.DataFrame, 
                         vis_files: List[Path], 
                         output_path: Path,
                         write_readme: bool = True):
    """Generate a detailed Markdown report of the analysis and optionally update README."""
    
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    # Analysis report
    lines = [
        f"# Conversation Analysis Report — {ts}",
        "",
        "## Overview",
        f"- **Total Messages:** {analysis['participation']['total_messages']}",
        f"- **Total Words:** {analysis['participation']['total_words']}",
        f"- **Messages per Turn:** {analysis['participation']['messages_per_turn']:.2f}",
        "",
        "## Participation Analysis",
        "",
        "### By Speaker",
        "| Speaker | Messages | Words | Avg Length |",
        "|---------|----------|-------|------------|"
    ]
    
    for speaker in analysis['participation']['by_speaker']['message_counts']:
        msgs = analysis['participation']['by_speaker']['message_counts'][speaker]
        words = analysis['participation']['by_speaker']['word_counts'][speaker]
        avg = analysis['participation']['by_speaker']['avg_length'][speaker]
        lines.append(f"| {speaker} | {msgs} | {words} | {avg:.1f} |")
    
    lines.extend([
        "",
        "## Interaction Patterns",
        "",
        "### Most Common Response Pairs",
        "| Pattern | Count |",
        "|---------|-------|"
    ])
    
    for pattern, count in analysis['interaction_patterns']['turn_taking']['most_common_pairs']:
        lines.append(f"| {pattern} | {count} |")
    
    lines.extend([
        "",
        "## Content Analysis",
        "",
        "### Focus Distribution",
        "| Category | Mentions |",
        "|----------|----------|",
        f"| Technical | {analysis['content']['content_focus']['technical']} |",
        f"| UX/Design | {analysis['content']['content_focus']['ux']} |",
        f"| Data/Analytics | {analysis['content']['content_focus']['data']} |",
    ])
    
    if vis_files:
        lines.extend([
            "",
            "## Visualizations",
            ""
        ])
        for f in vis_files:
            lines.append(f"![{f.stem}]({f.name})")
    
    lines.extend([
        "",
        "## Raw Statistics",
        "",
        "```json",
        json.dumps(analysis, indent=2),
        "```"
    ])
    
    output_path.write_text("\n".join(lines), encoding="utf-8")

def analyze_experiment_results(df: pd.DataFrame) -> Dict[str, Any]:
    """Additional analysis for experiment results with variants."""
    if 'variant' not in df.columns:
        return {}
    
    return {
        'by_variant': {
            variant: {
                'messages': len(variant_df),
                'avg_length': variant_df['word_count'].mean(),
                'speaker_distribution': variant_df['speaker'].value_counts().to_dict(),
                'content_focus': analyze_content(variant_df)['content_focus']
            }
            for variant, variant_df in df.groupby('variant')
        }
    }

def update_project_readme(analysis: Dict[str, Any], report_path: Path):
    """Update the project README.md with latest analysis summary."""
    readme_path = report_path.parents[1] / "README.md"
    if not readme_path.exists():
        return
    
    summary = {
        'messages': analysis['participation']['total_messages'],
        'words': analysis['participation']['total_words'],
        'avg_length': analysis['participation']['total_words'] / analysis['participation']['total_messages'],
        'top_pairs': analysis['interaction_patterns']['turn_taking']['most_common_pairs'][:3],
        'content_focus': analysis['content']['content_focus']
    }
    
    # Read existing README
    content = readme_path.read_text(encoding='utf-8').split('\n')
    
    # Find or create Latest Results section
    results_start = -1
    results_end = -1
    for i, line in enumerate(content):
        if line.startswith('## Latest Results'):
            results_start = i
        elif results_start > -1 and line.startswith('##'):
            results_end = i
            break
    
    if results_start == -1:
        # Add section at end
        content.extend([
            "",
            "## Latest Results",
            ""
        ])
        results_start = len(content) - 1
        results_end = len(content)
    
    # Generate new results section
    new_results = [
        "## Latest Results",
        "",
        f"Analysis from {datetime.now().strftime('%Y-%m-%d %H:%M')}:",
        "",
        "- **Messages:** {messages}".format(**summary),
        "- **Total Words:** {words}".format(**summary),
        "- **Average Length:** {avg_length:.1f} words".format(**summary),
        "",
        "**Top Interaction Patterns:**",
        "```",
        *[f"{pattern}: {count} exchanges" for pattern, count in summary['top_pairs']],
        "```",
        "",
        "**Content Focus:**",
        "```",
        f"Technical: {summary['content_focus']['technical']} mentions",
        f"UX/Design: {summary['content_focus']['ux']} mentions",
        f"Data/Analytics: {summary['content_focus']['data']} mentions",
        "```",
        "",
        f"See [conversation analysis]({report_path.relative_to(readme_path.parent)}) for details.",
        ""
    ]
    
    # Replace results section
    content[results_start:results_end] = new_results
    
    # Write updated README
    readme_path.write_text('\n'.join(content), encoding='utf-8')
    return readme_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs='?', type=Path, default=None,
                        help="Path to conversation JSONL file. If omitted, the most recent conversation log under the project's `outputs/` directory will be used.")
    parser.add_argument("--experiment", action="store_true", 
                      help="Input contains experiment results with variants")
    parser.add_argument("--output-dir", type=Path,
                      help="Output directory (default: same as input)")
    parser.add_argument("--skip-readme", action="store_true",
                      help="Skip updating the project README.md")
    args = parser.parse_args()
    
    # If input not provided, locate the most recent conversation log in the project's outputs directory
    if args.input is None:
        # project outputs directory is assumed to be one level up from the scripts folder: ../outputs
        script_dir = Path(__file__).resolve().parent
        project_dir = script_dir.parent
        outputs_dir = project_dir / 'outputs'

        candidates = []
        if outputs_dir.exists():
            # Preferred patterns in order
            for pattern in ("conversation_log*.jsonl", "conversation*.jsonl", "*.jsonl"):
                candidates.extend(list(outputs_dir.glob(pattern)))

        if not candidates:
            parser.error(f"No input provided and no conversation logs found in {outputs_dir!s}")

        # pick newest by modification time
        args.input = max(candidates, key=lambda p: p.stat().st_mtime)
        print(f"No input supplied — using most recent conversation log: {args.input}")

    if not args.output_dir:
        args.output_dir = args.input.parent

    # Load and analyze
    df = load_transcript(args.input)
    
    analysis = {
        'participation': analyze_participation(df),
        'interaction_patterns': analyze_interaction_patterns(df),
        'content': analyze_content(df)
    }
    
    if args.experiment:
        analysis['experiment'] = analyze_experiment_results(df)
    
    # Generate visualizations
    vis_dir = args.output_dir / "visualizations"
    vis_files = visualize_conversation(df, vis_dir)
    
    # Write reports
    report_path = args.output_dir / f"analysis_{args.input.stem}.md"
    write_analysis_report(analysis, df, vis_files, report_path)
    
    # Update README if requested
    if not args.skip_readme:
        try:
            readme_path = update_project_readme(analysis, report_path)
            if readme_path:
                print(f"Updated project README: {readme_path}")
        except Exception as e:
            print(f"Warning: Could not update README: {e}")
    
    print(f"Analysis complete. Report written to: {report_path}")
    if vis_files:
        print(f"Visualizations saved in: {vis_dir}")

if __name__ == "__main__":
    main()