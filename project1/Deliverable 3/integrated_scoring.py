# integrated_scoring.py
  #for deliverable 3
#End-to-end scoring helper that:
#1) Fetches & extracts page text automatically.
#2) Applies a tiny ML signal (transformers sentiment if available; VADER fallback).
#3) Calls your existing rules+NLP function, merges ML notes, and returns JSON.

#This wraps (but does not modify the Deliverable 1 scorer.


from typing import Optional, Dict, Any, Tuple
import re

from deliverable1 import evaluate_url_credibility, score_text_signals  # reuse your functions

# ---- 2) Automatic text fetch ----
def fetch_page_text(url: str, timeout: float = 4.0, max_chars: int = 20000) -> Tuple[Optional[str], str]:
    
    #Try multiple extractors, from best to simplest.
    #Returns (text or None, fetch_explanation).
    
    steps = []
    try:
        # We import lazily so your base module doesn't require these at install time.
        import requests
        headers = {"User-Agent": "credibility-scorer/0.1"}
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        html = resp.text
        steps.append("Fetched HTML via requests.")

        # Try trafilatura first (best for messy sites)
        try:
            import trafilatura
            extracted = trafilatura.extract(html, include_comments=False, include_tables=False)
            if extracted and extracted.strip():
                text = extracted.strip()
                steps.append("Extracted text with trafilatura.")
                return text[:max_chars], "; ".join(steps)
        except Exception:
            steps.append("trafilatura not available or failed.")

        # Try readability-lxml
        try:
            from readability import Document
            doc = Document(html)
            summary_html = doc.summary(html_partial=True)
            from bs4 import BeautifulSoup
            text = BeautifulSoup(summary_html, "lxml").get_text(" ", strip=True)
            if text and text.strip():
                steps.append("Extracted text with readability-lxml.")
                return text[:max_chars], "; ".join(steps)
        except Exception:
            steps.append("readability-lxml not available or failed.")

        # Fallback: raw BeautifulSoup text
        try:
            from bs4 import BeautifulSoup
            text = BeautifulSoup(html, "lxml").get_text(" ", strip=True)
            if text and text.strip():
                steps.append("Extracted text with BeautifulSoup fallback.")
                return text[:max_chars], "; ".join(steps)
        except Exception:
            steps.append("BeautifulSoup fallback not available or failed.")

        steps.append("No extractor produced text.")
        return None, "; ".join(steps)
    except Exception as e:
        steps.append(f"Fetch failed: {e}")
        return None, "; ".join(steps)

# ---- 3) ML signal (transformers sentiment ▷ VADER ▷ no-op) ----
def ml_text_signal(text: Optional[str]) -> Tuple[float, str]:
    
    #Returns (delta, note). Small, conservative contribution:
     # - strong negative sentiment → small negative delta
     # - strong positive sentiment → even smaller positive delta
    
    if not text or not text.strip():
        return 0.0, "ML: no text provided."

    # Try transformers pipeline
    try:
        from transformers import pipeline  # pip install transformers
        nlp = pipeline("sentiment-analysis")  # uses a small default model
        # Use first ~2000 chars for speed
        snippet = text[:2000]
        out = nlp(snippet[:512])[0]  # run once on a short snippet
        label = out["label"].upper()
        score = float(out["score"])
        # Map sentiment to a tiny credibility delta
        if label == "NEGATIVE":
            delta = round(-3.0 * score, 2)  # up to ~-3
        elif label == "POSITIVE":
            delta = round( 2.0 * score, 2)  # up to ~+2
        else:
            delta = 0.0
        return delta, f"ML(transformers): {label} ({score:.2f}) → {delta:+.2f}"
    except Exception:
        pass

    # Fallback: VADER (if available)
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer  # pip install nltk
        sia = SentimentIntensityAnalyzer()
        s = sia.polarity_scores(text[:4000])
        comp = s["compound"]
        delta = round(5.0*comp if comp > 0 else 7.0*comp, 2)  # ~[-7,+5], can be tuned
        # Clamp to smaller impact to keep rules dominant
        if delta > 2.0: delta = 2.0
        if delta < -3.0: delta = -3.0
        return float(delta), f"ML(VADER): compound={comp:+.3f} → {delta:+.2f}"
    except Exception:
        pass

    return 0.0, "ML: no ML backend available; skipped."

# ---- 4) Orchestrator ----
def score_url_with_fetch_and_ml(url: str, use_text: bool = True) -> Dict[str, Any]:
    
    #End-to-end helper:
      #- fetches page text
      #- (optionally) applies ML delta and notes
      #- calls your rules+NLP scorer with the fetched text
      #- merges explanations and returns a unified JSON

    #Output schema (superset of Deliverable 1):
    #{
    #  "score": float,
    #  "explanation": str,
    #  "extras": {
    #    "fetched": bool,
    #    "fetch_info": str,
    #    "ml": {"delta": float, "note": str},
    #    "chars_analyzed": int
    #  }
    #}
    
    text, fetch_info = fetch_page_text(url)
    fetched = text is not None
    ml_delta, ml_note = ml_text_signal(text) if fetched else (0.0, "ML skipped: no text.")

    # Call your existing scorer. If fetched and use_text=True, your NLP regex/lexicon will run.
    base = evaluate_url_credibility(
        url=url,
        page_text=text or "",
        use_text=use_text and fetched
    )

    # Merge ML delta into the final score and explanation (keep impact small)
    final_score = float(max(0.0, min(100.0, round(base["score"] + ml_delta, 1))))
    final_expl = base["explanation"] + f" {ml_note}"

    return {
        "score": final_score,
        "explanation": final_expl,
        "extras": {
            "fetched": fetched,
            "fetch_info": fetch_info,
            "ml": {"delta": float(ml_delta), "note": ml_note},
            "chars_analyzed": len(text) if text else 0
        }
    }

# ---- 5) Convenience CLI / quick test ----
if __name__ == "__main__":
    test_url = "https://www.nejm.org/doi/full/10.1056/NEJMoa2034577"
    result = score_url_with_fetch_and_ml(test_url, use_text=True)
    import json
    print(json.dumps(result, indent=2, ensure_ascii=False))
