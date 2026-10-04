import os
from typing import Optional, Tuple

import requests
import streamlit as st
from crewai import LLM
from streamlit.errors import StreamlitSecretNotFoundError

# ---------------------------------------------------------------------------
# Free AI providers you can choose in the app's sidebar.
# Model names and free limits change. If a model stops working, edit the
# "models" lists below (left side = what you see, right side = the real ID).
# Gemini 2.5 models are intentionally NOT listed (being shut down in Oct 2026).
# ---------------------------------------------------------------------------
PROVIDERS = {
    "Gemini (Google)": {
        "kind": "gemini",
        "env": "GEMINI_API_KEY",
        "key_url": "https://aistudio.google.com/apikey",
        "models": {
            "Gemini 3.8 Flash": "gemini-3.8-flash",
            "Gemini 3.5 Flash-Lite (lighter)": "gemini-3.5-flash-lite",
        },
    },
    "Groq (free, very fast)": {
        "kind": "openai_compatible",
        "env": "GROQ_API_KEY",
        "key_url": "https://console.groq.com/keys",
        "base_url": "https://api.groq.com/openai/v1",
        "models": {
            "OpenAI gpt-oss-120b (open model)": "openai/gpt-oss-120b",
            "Meta Llama 3.3 70B": "llama-3.3-70b-versatile",
        },
    },
    "OpenRouter (free models)": {
        "kind": "openai_compatible",
        "env": "OPENROUTER_API_KEY",
        "key_url": "https://openrouter.ai/keys",
        "base_url": "https://openrouter.ai/api/v1",
        "models": {
            "OpenAI gpt-oss-120b (free)": "openai/gpt-oss-120b:free",
            "Meta Llama 3.3 70B (free)": "meta-llama/llama-3.3-70b-instruct:free",
        },
    },
}
DEFAULT_PROVIDER = "Gemini (Google)"


def clean_key(raw: Optional[str]) -> Optional[str]:
    """Remove spaces, new lines and quote marks that sneak in when pasting a key."""
    if not raw:
        return None
    key = raw.strip().strip("'\"").strip()
    return key or None


def resolve_api_key(provider: str, typed_key: Optional[str] = None) -> Optional[str]:
    env_name = PROVIDERS[provider]["env"]
    key = clean_key(typed_key) or clean_key(os.getenv(env_name))
    if not key:
        try:
            key = clean_key(st.secrets.get(env_name))
        except (StreamlitSecretNotFoundError, FileNotFoundError):
            key = None
    return key


def setup_environment(provider: str, typed_key: Optional[str] = None) -> bool:
    """Store the chosen provider's key and clear leftovers that cause 401 errors."""
    key = resolve_api_key(provider, typed_key)
    if not key:
        return False

    # Leftover OpenAI-style or Google login/Vertex settings can make the key get
    # sent the wrong way. Clear them; the key is passed to the model directly.
    for var in (
        "OPENAI_API_BASE",
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_MODEL_NAME",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_GENAI_USE_VERTEXAI",
    ):
        os.environ.pop(var, None)

    os.environ[PROVIDERS[provider]["env"]] = key
    if PROVIDERS[provider]["kind"] == "gemini":
        # A stale GOOGLE_API_KEY (e.g. from Colab) would override ours, so match it.
        os.environ["GOOGLE_API_KEY"] = key
    return True


def get_llm(provider: str, model_id: str, temperature: float = 0.3) -> LLM:
    """Build the model object all agents use. No LiteLLM needed (native CrewAI only)."""
    info = PROVIDERS[provider]
    key = os.environ.get(info["env"])

    if info["kind"] == "gemini":
        return LLM(model=f"gemini/{model_id}", api_key=key, temperature=temperature)

    def make(name: str) -> LLM:
        return LLM(
            model=name,
            custom_openai=True,
            base_url=info["base_url"],
            api_key=key,
            temperature=temperature,
        )

    llm = make(model_id)
    # CrewAI removes a leading "openai/" from the model name. Some real model IDs
    # start with "openai/" (e.g. openai/gpt-oss-120b), so add it back if it was removed.
    sent = getattr(llm, "model", None)
    if model_id.startswith("openai/") and isinstance(sent, str) and sent != model_id:
        llm = make("openai/" + model_id)
    return llm


def test_api_key(
    provider: str, model_id: str, typed_key: Optional[str] = None
) -> Tuple[bool, str]:
    """Send one tiny request straight to the provider to check the key and model."""
    info = PROVIDERS[provider]
    key = resolve_api_key(provider, typed_key)
    if not key:
        return False, "No API key found. Paste your key first."

    try:
        if info["kind"] == "gemini":
            response = requests.post(
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model_id}:generateContent",
                headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                json={"contents": [{"parts": [{"text": "Reply with the word: ok"}]}]},
                timeout=30,
            )
        else:
            response = requests.post(
                f"{info['base_url']}/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={
                    "model": model_id,
                    "messages": [{"role": "user", "content": "Reply with the word: ok"}],
                    "max_tokens": 20,
                },
                timeout=30,
            )
    except requests.RequestException as error:
        return False, f"Could not reach the provider: {error}"

    code = response.status_code
    if code == 200:
        return True, f"Key works with {model_id}."
    if code == 401:
        return False, (
            f"401: the provider rejected the key. Create a fresh key at {info['key_url']} "
            "and paste only the key (no quotes or spaces)."
        )
    if code == 403:
        return False, "403: this key is not allowed to use this model or service."
    if code == 404:
        return False, f"404: model '{model_id}' was not found or has been retired. Pick another model."
    if code == 429:
        return False, "429: free limit reached for now. Wait a bit, or switch provider or model."
    return False, f"Unexpected response {code}: {response.text[:300]}"
