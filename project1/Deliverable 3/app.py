# app.py
# Final Space app with output-mode toggle:
# - Default: "Score & Stars" (big number + star rating)
# - Optional: "JSON" (full detailed payload for transparency/debugging)

import json
import time
import gradio as gr

from deliverable1 import evaluate_url_credibility

# Optional advanced helpers (auto-fetch + ML tone). Degrade gracefully if missing.
HAVE_INTEGRATED = True
try:
    from integrated_scoring import fetch_page_text, ml_text_signal
except Exception:
    HAVE_INTEGRATED = False


def _score_with_all_features(url, page_text, use_text, auto_fetch_text, enable_ml):
    """End-to-end pipeline with optional text fetching and ML tone adjustment."""
    t0 = time.perf_counter()
    url = (url or "").strip()
    text = (page_text or "").strip()

    if not url:
        return {"score": 0.0, "explanation": "error: Empty URL string."}

    # --- Auto-fetch main article text if requested and none provided ---
    fetched = False
    fetch_info = "no fetch attempted"
    if auto_fetch_text and not text:
        if HAVE_INTEGRATED:
            text, fetch_info = fetch_page_text(url)
            fetched = text is not None
            text = text or ""
        else:
            fetch_info = "auto_fetch_text requested but integrated_scoring not available"

    # --- Core rules + optional NLP on text ---
    base = evaluate_url_credibility(
        url=url,
        page_text=text if (use_text and text) else None,
        use_text=bool(use_text and text),
    )

    # --- Small ML tone adjustment (only if text analyzed) ---
    ml_delta, ml_note = 0.0, "ML skipped"
    if enable_ml and HAVE_INTEGRATED and (use_text and text):
        try:
            ml_delta, ml_note = ml_text_signal(text)
            base["score"] = float(max(0.0, min(100.0, round(base["score"] + ml_delta, 1))))
            base["explanation"] = f"{base['explanation']} {ml_note}"
        except Exception as e:
            ml_note = f"ML error: {e}"

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    return {
        "score": base["score"],
        "explanation": base["explanation"],
        "extras": {
            "latency_ms": latency_ms,
            "used_text": bool(use_text and text),
            "auto_fetched": fetched,
            "fetch_info": fetch_info,
            "ml": {"delta": ml_delta, "note": ml_note},
            "chars_analyzed": len(text),
        },
    }


def _stars_for_score(score: float) -> str:
    """Map 0–100 to 1–5 star glyphs (rounded)."""
    n = max(0, min(5, int(round((score or 0.0) / 20.0))))
    return "★" * n + "☆" * (5 - n)


def predict(url, page_text, use_text, auto_fetch_text, enable_ml, output_mode):
    """Gradio API entrypoint: returns UI components depending on output mode."""
    res = _score_with_all_features(url, page_text, use_text, auto_fetch_text, enable_ml)
    score = float(res.get("score", 0.0))
    stars = _stars_for_score(score)
    pretty_json = json.dumps(res, indent=2, ensure_ascii=False)

    if output_mode == "Score & Stars":
        return (
            gr.update(value=f"{score:.1f}", visible=True),
            gr.update(value=stars, visible=True),
            gr.update(value=pretty_json, visible=False),
        )
    else:  # JSON
        return (
            gr.update(value="", visible=False),
            gr.update(value="", visible=False),
            gr.update(value=pretty_json, visible=True),
        )


DESCRIPTION = """
### Credibility Scorer (Hybrid: URL Heuristics + NLP + ML)
- **URL heuristics:** HTTPS, TLD/domain reputation, path/query/length signals.  
- **NLP (optional):** Sensational & weasel terms, ALL-CAPS, punctuation, DOI.  
- **ML (optional):** Small tone adjustment (Transformers → VADER fallback).  
- **Auto-fetch (optional):** Extract main article text when none is provided.  
Use the **Output Mode** switch to show either a big numeric score with stars (default) or the detailed JSON.
"""

with gr.Blocks(title="Credibility Scorer", theme="soft") as demo:
    gr.Markdown(DESCRIPTION)

    url = gr.Textbox(label="URL", placeholder="https://example.com/article", lines=1)
    page_text = gr.Textbox(label="Optional: Article Text (paste to control exactly what is analyzed)", lines=8)

    with gr.Row():
        use_text = gr.Checkbox(label="Use NLP on text", value=True)
        auto_fetch_text = gr.Checkbox(label="Auto-fetch text when empty", value=True)
        enable_ml = gr.Checkbox(label="Enable ML tone adjustment", value=True)

    output_mode = gr.Radio(
        ["Score & Stars", "JSON"],
        label="Output Mode",
        value="Score & Stars",  # default
    )

    run = gr.Button("Score")

    # Outputs: two textboxes for score/stars, and a code block for JSON.
    with gr.Row():
        score_display = gr.Textbox(label="Final Score (0–100)", interactive=False, visible=True)
        star_display = gr.Textbox(label="Rating (★)", interactive=False, visible=True)

    result_json = gr.Code(label="Detailed Output (JSON)", language="json", visible=False)

    run.click(
        fn=predict,
        inputs=[url, page_text, use_text, auto_fetch_text, enable_ml, output_mode],
        outputs=[score_display, star_display, result_json],
        api_name="predict",
    )

if __name__ == "__main__":
    demo.launch()
