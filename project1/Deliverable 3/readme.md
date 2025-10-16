# Credibility Scorer — Deliverable 3

This repository contains the final **production-ready version** of the Credibility Scorer project.  
It is fully deployed as a live application on **Hugging Face Spaces**.

🔗 **Live Demo:** [https://huggingface.co/spaces/VonVan/Credibility-Scorer](https://huggingface.co/spaces/VonVan/Credibility-Scorer)

---

## Overview

The **Credibility Scorer** evaluates the trustworthiness of online articles in real time.  
It uses a hybrid pipeline combining:

- **Rules-based analysis:** Domain authority, HTTPS, TLD quality, and URL structure  
- **NLP signals:** Detection of sensational or vague language  
- **ML tone adjustment:** Sentiment-based confidence calibration  
- **Auto-fetching:** Optional text extraction from URLs  

The deployed model runs on **Gradio**, providing both a web interface and an API endpoint.

---

## 🚀 Deployment Demonstration

The project meets the “Production-Ready Pipeline” requirement through deployment on **Hugging Face Spaces**.  
The following Python code demonstrates the deployment process used to publish this model:

```python
from huggingface_hub import HfApi, whoami, upload_folder

HF_TOKEN = "hf_..."  # Replace with your Hugging Face write token
api = HfApi(token=HF_TOKEN)
user = whoami(token=HF_TOKEN)["name"]
space_id = f"{user}/credibility-scorer"

api.create_repo(repo_id=space_id, repo_type="space", space_sdk="gradio", exist_ok=True)
upload_folder(repo_id=space_id, repo_type="space", folder_path=".", path_in_repo=".")
print(f"Deployed to https://huggingface.co/spaces/{space_id}")