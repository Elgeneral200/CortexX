# src/chatbot/engine.py

import os
from typing import List, Dict

import requests

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"  # adjust if you prefer another Groq model


SYSTEM_PROMPT = """
You are CortexX AI, an expert assistant embedded in an enterprise sales forecasting
and demand planning platform called CortexX-Forecasting.

Your responsibilities:
- Explain data quality metrics, filters, KPIs, dashboards, and charts.
- Help users understand forecasting models (e.g., XGBoost, LightGBM, Random Forest, Prophet),
  hyperparameter tuning, backtesting, and prediction intervals.
- Provide clear, concise, business-focused explanations suitable for analysts and managers.
- When you reference actions in the app, describe what the user should click or check,
  but never claim to have direct access to their data.

If a question is unrelated to analytics, ML, or the platform, answer briefly and gently
guide the conversation back to sales forecasting and the CortexX app.
""".strip()


def _build_messages(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    # history is already [{'role': 'user'|'assistant', 'content': str}, ...]
    messages.extend(history)
    return messages


def generate_reply(history: List[Dict[str, str]]) -> str:
    """
    Generate a reply from Groq using chat completions API.
    history: list of messages with 'role' and 'content'.
    Returns assistant text; on failure returns a safe fallback.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return (
            "CortexX AI is not configured yet because GROQ_API_KEY "
            "is missing in the environment."
        )

    try:
        payload = {
            "model": GROQ_MODEL,
            "messages": _build_messages(history),
            "temperature": 0.2,
            "max_tokens": 512,
        }

        response = requests.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        return (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
            or "I could not generate a response. Please try rephrasing your question."
        )
    except Exception as exc:
        return (
            "CortexX AI encountered an error while contacting the Groq service. "
            f"Details for debugging: {type(exc).__name__}."
        )
