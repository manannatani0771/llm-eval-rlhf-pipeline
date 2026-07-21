# LLM Response Evaluation & RLHF Data Pipeline

A complete data pipeline demonstrating the core stages of RLHF (Reinforcement Learning from Human Feedback) data collection — from multi-response generation through human preference labeling to QA-validated, export-ready preference pairs.

## Overview

This project simulates the annotation and quality-control workflow used to build preference datasets for training reward models in RLHF pipelines. It bridges data operations / annotation QA experience with practical GenAI engineering.

## Pipeline Stages

1. **Response Generation** — For each prompt, two responses are generated via the Groq API (Llama 3.1 8B) at different temperatures (0.3 = focused, 0.9 = creative), simulating the multi-candidate generation step needed before human ranking.

2. **Human Annotation UI** — A Streamlit interface presents each prompt with both responses side by side. The annotator selects the better response (or marks a tie) and provides a written rationale — mirroring structured annotation QA workflows (entity/sentiment/classification tasks) used in production ML data pipelines.

3. **QA & Analysis** — A metrics script analyzes collected labels: total judgments, tie rate, win-rate breakdown by generation temperature, and sample rationale review.

4. **Preference Pair Export** — Labeled data is exported in the standard `chosen/rejected` format used to train reward models in real RLHF systems (e.g., InstructGPT, Anthropic's HH-RLHF format).

## Tech Stack

- **Python 3.9**
- **Streamlit** — annotation interface
- **SQLite** — response and label storage
- **Groq API** (Llama 3.1 8B Instant) — response generation
- **Pandas** — data analysis and export

## Project Structure## How to Run

```bash
# 1. Set up environment
python3 -m venv venv
source venv/bin/activate
pip install streamlit pandas scikit-learn groq python-dotenv

# 2. Add your Groq API key
echo "GROQ_API_KEY=your_key_here" >> .env

# 3. Generate responses
python3 src/generate_responses.py

# 4. Launch annotation UI
streamlit run src/annotate_app.py

# 5. Run QA analysis and export
python3 src/qa_metrics.py
```

## Why This Matters

Reward model quality in RLHF depends entirely on the quality of human preference data underneath it. This project demonstrates the often-overlooked but critical layer of that pipeline: structured annotation, clear labeling rationale, and QA validation — skills directly transferable from data annotation and quality operations into GenAI/LLM evaluation roles.

## Future Extensions

- Scale to 100+ prompts and multiple annotators to compute inter-annotator agreement (Cohen's Kappa)
- Train a lightweight reward model (e.g., DistilBERT classifier) on the exported preference pairs
- Add automated LLM-as-judge scoring to compare against human labels

## Author

Manan Natani 
