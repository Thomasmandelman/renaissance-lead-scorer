"""
secrets.py — small helper to read API keys / DB credentials.

Reading order:
  1. Streamlit secrets (st.secrets[...]) — used when deployed on Streamlit
     Community Cloud, which sets a TOML-based secrets file.
  2. Process environment (os.environ[...]) — used when running locally
     where python-dotenv loaded a .env file at startup.

Both `app.py` and `db.py` import from here so that env-var access logic
lives in one place and stays consistent between local and deployed.
"""
import os
from typing import Optional


def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Look up a credential by name. Try Streamlit secrets first (which works
    only inside a Streamlit runtime), then fall back to os.environ.

    Returns the default if the key is not found in either source.
    """
    # Streamlit's secrets API is only meaningful when we're actually running
    # inside Streamlit. Importing it eagerly is fine — it is part of the
    # `streamlit` package which is already a dependency.
    try:
        import streamlit as st  # type: ignore
        # st.secrets behaves like a dict but raises if the key is missing.
        # Catch both the "not running under streamlit" case and the
        # "key absent" case and continue to the env-var fallback.
        if hasattr(st, "secrets"):
            try:
                value = st.secrets[key]
                if value is not None and value != "":
                    return str(value)
            except (KeyError, FileNotFoundError, Exception):
                pass
    except ImportError:
        pass

    # Local development path: read from environment (populated from .env
    # by python-dotenv at process start).
    return os.environ.get(key, default)


def require_secret(key: str) -> str:
    """
    Same as get_secret, but raises a clear error if the key is missing.
    Use this for credentials the app cannot run without (e.g. Supabase URL).
    """
    val = get_secret(key)
    if val is None or val == "":
        raise RuntimeError(
            f"Missing required secret: {key}. "
            f"Set it in Streamlit Cloud's Secrets panel "
            f"or in your local .env file."
        )
    return val