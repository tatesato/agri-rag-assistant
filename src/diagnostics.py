"""Opt-in development diagnostics with credential redaction."""
import os
from pathlib import Path
import re
import traceback

from dotenv import dotenv_values


def safe_traceback(exc: Exception) -> str:
    """Preserve chained exceptions without dumping locals or API credentials."""
    text = "".join(traceback.TracebackException.from_exception(
        exc, capture_locals=False).format())
    values = dict(dotenv_values(Path(__file__).resolve().parents[1] / ".env"))
    secrets = [value for key, value in [*values.items(), *os.environ.items()]
               if value and any(word in key.upper() for word in ("KEY", "TOKEN", "SECRET", "PASSWORD"))]
    for secret in sorted(secrets, key=len, reverse=True):
        text = text.replace(secret, "[REDACTED]")
    return re.sub(r"sk-[A-Za-z0-9_*-]+", "[REDACTED]", text)
