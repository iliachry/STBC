import os
from pathlib import Path


def load_dotenv(env_path: str | None = None) -> None:
    """Load key=value pairs from a .env file into os.environ (non-destructive for existing keys)."""
    path = Path(env_path) if env_path else Path.cwd() / ".env"
    if not path.exists():
        return
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                os.environ.setdefault(key, value)
    except Exception:
        # Do not fail app startup if .env parsing fails
        pass 