"""Environment and application configuration.

Loaded across all phases. Phase 0 only documents expected variables;
later phases will add typed settings (API keys, model names, paths, etc.).
"""

import os

from dotenv import load_dotenv

load_dotenv()


def get_settings() -> dict:
    """Return application settings from environment variables.

    TODO: expand as each phase introduces real config keys.
    """
    return {
        "groq_api_key": os.getenv("GROQ_API_KEY", ""),
    }
