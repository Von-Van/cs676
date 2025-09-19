# app.py
# For Deliverable 3!
from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl
from typing import Optional
import uvicorn
from deliverable1 import evaluate_url_credibility

app = FastAPI(title="Credibility Scorer (HF Space)")

class ScoreRequest(BaseModel):
    url: HttpUrl
    page_text: Optional[str] = None
    use_text: bool = False

class ScoreResponse(BaseModel):
    score: float
    explanation: str

@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    out = evaluate_url_credibility(str(req.url), page_text=req.page_text, use_text=req.use_text)
    return ScoreResponse(**out)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
