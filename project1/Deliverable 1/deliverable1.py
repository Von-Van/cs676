#goal is to combine rules-based URL heuristics with a NLP layer (lexicons + regexes) to produce a credibility score for URLs (health/science focus).

from urllib.parse import urlparse
import re
import time
from statistics import mean, stdev

# Config (URL heuristics only)

REPUTABLE = {
    "nih.gov": 0.95, "cdc.gov": 0.95, "who.int": 0.95, "un.org": 0.9,
    "nejm.org": 0.95, "thelancet.com": 0.95, "bmj.com": 0.9,
    "nature.com": 0.9, "sciencedirect.com": 0.85,
    "nasa.gov": 0.9, "noaa.gov": 0.9, "doi.org": 0.85,
    "pubmed.ncbi.nlm.nih.gov": 0.95, "arxiv.org": 0.75, "ssrn.com": 0.75,
    "wikipedia.org": 0.7
}
HIGH_TRUST_TLDS = {".gov", ".edu", ".ac.uk", ".int"}
LOW_TRUST_TLDS  = {".xyz", ".click", ".info", ".top", ".zip", ".mov"}
POS_PATH_HINTS  = ["/research", "/journal", "/paper", "/study", "/publication", "/doi/"]
NEG_PATH_HINTS  = ["/opinion", "/editorial", "/sponsored", "/advertorial", "/press-release", "/promo"]
LOW_TRUST_HOST_HINTS = ["click", "buzz", "viral", "giveaway", "free-", "loan", "sweepstake", "casino", "bet", "adult", "porn", "nsfw"]

HOST_RE = re.compile(r"^[A-Za-z0-9.-]+$")


# Optional NLP layer (unused by default)

DEFAULT_NLP_CONFIG = {
    "lexicons": {
        "sensational": {"weight": -2, "terms": ["shocking","unbelievable","exposed","miracle","cure","scandal","you won't believe"]},
        "weasel": {"weight": -1, "terms": ["some say","experts claim","reportedly","allegedly","may","might","could"]},
        "subjective_pos": {"weight": 0.5, "terms": ["amazing","incredible","fantastic","remarkable","breakthrough"]},
        "subjective_neg": {"weight": -1, "terms": ["disaster","fraud","hoax","dangerous","toxic"]}
    },
    "patterns": {
        "all_caps": {"weight": -3, "regex": r"\b[A-Z]{4,}\b", "threshold": 3},
        "excess_punct": {"weight": -2, "regex": r"([!?])\1{2,}", "threshold": 2},
        "doi": {"weight": 2, "regex": r"\b10\.[0-9]{4,9}/[-._;()/:A-Z0-9]+\b", "i": True, "threshold": 1},
        "many_numbers_no_doi": {"weight": -3, "regex": r"\b\d+(?:\.\d+)?%?\b", "threshold": 5, "paired_negative_if": "doi"}
    },
    "delta_min": -15.0,
    "delta_max": 8.0
}

def _compile_pattern(pat: dict):
    flags = re.I if pat.get("i") else 0
    return re.compile(pat["regex"], flags)

def score_text_signals(page_text: str, nlp_config: dict = None):
    
    #Returns (delta_score, notes) using a tiny lexicon+regex approach.
    #Planned expansion: more lexicons, ML model in final deliverable. See integrated_scoring.py in repo for current preview
    #Designed to be fast and explainable. Default clamp ~[-15, +8].
    
    cfg = nlp_config or DEFAULT_NLP_CONFIG
    if not isinstance(page_text, str) or not page_text.strip():
        return 0.0, ["NLP skipped: no text provided."]

    notes = []
    t = page_text
    tl = t.lower()

    delta = 0.0
    # Lexicon counts
    for label, entry in cfg.get("lexicons", {}).items():
        w = float(entry.get("weight", 0))
        terms = entry.get("terms", [])
        count = sum(1 for term in terms if term in tl)
        if count:
            shift = w * min(count, 10)
            delta += shift
            notes.append(f"{label} terms ({count}) ({shift:+}).")

    # Patterns w/ thresholds and optional pairing
    compiled = {k: _compile_pattern(v) for k, v in cfg.get("patterns", {}).items()}
    hits_map = {k: len(rx.findall(t)) for k, rx in compiled.items()}

    for label, p in cfg.get("patterns", {}).items():
        hits = hits_map.get(label, 0)
        thr = int(p.get("threshold", 1))
        if hits >= thr:
            paired = p.get("paired_negative_if")
            if paired and hits_map.get(paired, 0) >= int(cfg["patterns"].get(paired, {}).get("threshold", 1)):
                # Skip penalty if the paired positive signal exists
                continue
            w = float(p.get("weight", 0))
            delta += w
            notes.append(f"{label} pattern (hits={hits}>=thr={thr}) ({w:+}).")

    # Clamp
    lo = float(cfg.get("delta_min", -15.0)); hi = float(cfg.get("delta_max", 8.0))
    delta = max(lo, min(hi, round(delta, 2)))
    return delta, notes


# URL scoring

def _domain_match(domain: str, pattern: str) -> bool:
    return domain == pattern or domain.endswith("." + pattern)

def evaluate_url_credibility(url: str, page_text: str = None, use_text: bool = False, nlp_config: dict = None) -> dict:
    
    #Proof-of-concept rules-based scorer with optional NLP hook (disabled by default).
    #Returns: {"score": float (0..100), "explanation": str}
    
    explanation = []
    try:
        if not isinstance(url, str) or not url.strip():
            raise ValueError("Empty URL string.")
        candidate = url.strip()
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://", candidate):
            candidate = "https://" + candidate
            explanation.append("No scheme; assumed HTTPS.")
        parsed = urlparse(candidate)
        if not parsed.netloc:
            raise ValueError("URL missing hostname.")
        host = parsed.netloc.lower()
        host = re.sub(r":\d+$", "", host)
        if (not HOST_RE.match(host)) or (host.count(".") < 1):
            raise ValueError("Invalid hostname.")
        path  = parsed.path or "/"
        query = parsed.query or ""

        score = 50.0

        # Scheme
        if parsed.scheme.lower() == "https":
            score += 5;  explanation.append("Uses HTTPS (+5).")
        elif parsed.scheme.lower() == "http":
            score -= 10; explanation.append("Uses HTTP (−10).")

        # Known reputable domains
        bonus = 0.0; matched = None
        for dom, w in REPUTABLE.items():
            if _domain_match(host, dom):
                b = 20 * w
                if b > bonus: bonus, matched = b, dom
        if matched:
            score += bonus; explanation.append(f"Recognized reputable domain '{matched}' (+{bonus:.1f}).")

        # TLD
        tld = "." + host.split(".")[-1]
        if any(host.endswith(t) for t in HIGH_TRUST_TLDS) or (tld in HIGH_TRUST_TLDS):
            score += 10; explanation.append(f"High-trust TLD ({tld}) (+10).")
        elif tld in LOW_TRUST_TLDS:
            score -= 10; explanation.append(f"Low-trust TLD ({tld}) (−10).")

        # Path hints
        path_l = path.lower()
        if any(h in path_l for h in POS_PATH_HINTS):
            score += 5; explanation.append("Research-oriented path hint (+5).")
        neg_hits = [h for h in NEG_PATH_HINTS if h in path_l]
        if neg_hits:
            score -= 8; explanation.append(f"Promotional/opinion path hint {neg_hits[0]} (−8).")

        # Host lexical hints
        if any(h in host for h in LOW_TRUST_HOST_HINTS):
            score -= 10; explanation.append("Low-trust lexical hint in host (−10).")

        # Query complexity
        if query:
            n_params = query.count("&") + 1
            if n_params >= 4:
                score -= 3; explanation.append("Many tracking parameters (−3).")

        # URL length
        L = len(candidate)
        if L > 180:
            score -= 5; explanation.append("Very long URL (−5).")
        elif L > 120:
            score -= 2; explanation.append("Long URL (−2).")

        # Optional NLP layer (currently disabled unless use_text=True)
        if use_text:
            delta, notes = score_text_signals(page_text or "", nlp_config=nlp_config)
            score += delta
            explanation.extend(notes)
        else:
            explanation.append("NLP layer present but disabled (set use_text=True to enable).")

        score = float(max(0.0, min(100.0, round(score, 1))))
        if not explanation:
            explanation = ["No strong signals; neutral baseline."]
        return {"score": score, "explanation": " ".join(explanation)}
    except Exception as e:
        return {"score": 0.0, "explanation": f"error: {str(e)}"}


# Tests & micro-benchmark

def _assert(cond, msg):
    if not cond: raise AssertionError(msg)

TEST_URLS = [
    "https://www.nejm.org/doi/full/10.1056/NEJMoa2034577",
    "who.int/news-room/fact-sheets/detail/diabetes",
    "http://nasa.gov/some/path",
    "medium.com/@someone/opinion-on-vaccines-12345",
    "example.xyz/win-a-free-phone?utm_source=a&x=1&y=2&z=3&q=4",
    "https://subdomain.cdc.gov/research/article?id=1",
    "invalid-url-without-host",
    "",
]

# Evaluate (URL-only, NLP disabled)
results = [evaluate_url_credibility(u) for u in TEST_URLS]

# Schema checks
for r in results:
    _assert(set(r.keys()) == {"score","explanation"}, "Result must have exactly 'score' and 'explanation'")
    _assert(isinstance(r["score"], float), "Score must be float")
    _assert(isinstance(r["explanation"], str), "Explanation must be str")

# Basic ordering
good = evaluate_url_credibility("https://pubmed.ncbi.nlm.nih.gov/123456/")["score"]
bad  = evaluate_url_credibility("http://freegift.click/win")["score"]
_assert(good > bad, "Expected reputable domain to score higher than low-trust domain.")

# Malformed
bad1 = evaluate_url_credibility("")
bad2 = evaluate_url_credibility("not a url at all")
_assert(bad1["score"] == 0.0 and bad1["explanation"].startswith("error"), "Empty string should error")
_assert(bad2["score"] == 0.0 and bad2["explanation"].startswith("error"), "Malformed hostname should error")

# Optional: quick NLP check (disabled by default,  enable to prove works)
sample_text = "Some say this could be a miracle cure!!! DOI: 10.1016/j.cell.2024.01.001"
with_nlp = evaluate_url_credibility("https://example.com/research/article", page_text=sample_text, use_text=True)
_assert(with_nlp["score"] != evaluate_url_credibility("https://example.com/research/article")["score"], "NLP delta should alter score when enabled")

# Benchmark
def benchmark(n=5000, use_text=False):
    sample = [
        ("https://www.nejm.org/doi/full/10.1056/NEJMoa2034577", ""),
        ("https://doi.org/10.1016/j.cell.2023.01.001", ""),
        ("http://freegift.click/win", ""),
        ("https://wikipedia.org/wiki/Credibility", ""),
        ("https://example.com/opinion/politics?utm_source=a&x=1&y=2&z=3", "Some say this could be shocking!!!")  # small text for NLP path
    ]
    t0 = time.perf_counter()
    scores = []
    for i in range(n):
        u, txt = sample[i % len(sample)]
        r = evaluate_url_credibility(u, page_text=txt, use_text=use_text)
        scores.append(r["score"])
    t1 = time.perf_counter()
    return {
        "n": n,
        "use_text": use_text,
        "avg_score": mean(scores),
        "stdev": stdev(scores) if len(scores) > 1 else 0.0,
        "elapsed_s": round(t1 - t0, 4),
        "avg_us_per_call": round((t1 - t0) * 1e6 / n, 2)
    }

bench_url_only = benchmark(5000, use_text=False)
bench_with_nlp = benchmark(5000, use_text=True)

print("Sample outputs:")
for u, r in zip(TEST_URLS, results):
    print(" ", u, "->", r)
print("\nBenchmark (URL-only):", bench_url_only)
print("Benchmark (URL+NLP hooks ON):", bench_with_nlp)