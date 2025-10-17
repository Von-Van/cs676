# Conversation Quality Report

Heuristic metrics for realism, persona consistency, and response diversity.
Confidence is the assistant's self-reported 0–1 score (averaged).

| Persona | Avg Len (w) | ?-Ratio | TTR | Repeat Rate | Avg Lat (s) | Avg Conf | Realism | Consistency | Diversity | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Curious Student | 2.0 | 0.00 | 0.50 | 0.500 | 101.24 | — | 0.04 | 1.50 | 2.50 | noticeable repetition; very short answers |
| Busy Parent | 2.0 | 0.00 | 0.50 | 0.500 | 71.26 | — | 0.04 | 1.50 | 2.50 | noticeable repetition; very short answers |
| Small Business Owner | 2.0 | 0.00 | 0.50 | 0.500 | 71.76 | — | 0.04 | 1.50 | 2.50 | noticeable repetition; very short answers |

## Interpretation & Next Steps
- Realism favors medium-length, low-repetition answers with ~20% questions.
- Consistency improves when trait/goal/style terms are present and answer lengths are stable.
- Diversity rewards wider vocabulary and low repeated phrasing.
