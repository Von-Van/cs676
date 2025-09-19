import json
from typing import List, Dict, Any
from deliverable1 import evaluate_url_credibility  # my module

def write_deliverable1_test_json(output_path: str = "deliverable1_test_output.json") -> str:
    
    #Runs the two proof-of-concept tests and writes a JSON file.

    #Returns
    #-------
    #str
        #The path to the JSON file that was written.
        
    # Test 1: URL-only, NLP disabled (default)
    url_1 = "https://pubmed.ncbi.nlm.nih.gov/123456/"
    res_1 = evaluate_url_credibility(url_1)

    # Test 2: Same URL structure style, NLP hook enabled with sample text
    url_2 = "https://example.com/research/article"
    text_2 = "Some say this could be a miracle cure!!! DOI: 10.1016/j.cell.2024.01.001"
    res_2 = evaluate_url_credibility(url_2, page_text=text_2, use_text=True)

    # Assemble a clean, reproducible JSON payload
    payload: List[Dict[str, Any]] = [
        {
            "url": url_1,
            "use_text": False,
            "page_text_preview": None,
            "result": res_1  # {"score": float, "explanation": str}
        },
        {
            "url": url_2,
            "use_text": True,
            "page_text_preview": text_2[:160],
            "result": res_2  # {"score": float, "explanation": str}
        }
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return output_path

if __name__ == "__main__":
    path = write_deliverable1_test_json()
    print(f"Wrote test results to {path}")
