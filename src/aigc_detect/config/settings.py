"""Environment-derived settings, read once at import.

Everything the project needs from outside the repo lands here: credentials,
where the large directories live, which device to run on, which checkpoint to
serve. Nothing else in the codebase reads ``os.environ`` directly, so there is
one place to look when a value is not what you expected.

Precedence is the conventional one: a real environment variable wins over
``.env``, which wins over the default. That ordering matters for CI and for
one-off overrides -- ``AIGC_DEVICE=cpu uv run aigc embed ...`` has to beat
whatever ``.env`` says without editing the file.

DELIBERATELY CHEAP TO IMPORT. ``aigc_detect.config`` is imported by every
module and by every CLI invocation, including ``--help``. Nothing here may
import torch or touch the network: resolving the device costs a CUDA probe, so
that is a function (:func:`resolve_device`) rather than a field, and it is
called only by code that is about to use a GPU anyway.

See ``.env.example`` at the repo root for the documented variable list.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# src/aigc_detect/config/settings.py -> config -> aigc_detect -> src -> repo root
ROOT_DIR = Path(__file__).resolve().parents[3]

# override=False is the load-bearing half of the precedence rule above: a
# variable already exported in the shell is left alone.
load_dotenv(ROOT_DIR / ".env", override=False)


def _env(name: str) -> str | None:
    """Read an environment variable, treating blank as unset.

    ``.env.example`` ships every key present but empty (``HF_TOKEN=``), so a
    copied-and-unedited file would otherwise hand every consumer an empty
    string instead of None -- which reads as "a token was configured" and fails
    much later, at the API call, instead of here.
    """
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _env_path(name: str, default: Path) -> Path:
    """Resolve a path-valued variable, relative paths against the repo root."""
    raw = _env(name)
    if raw is None:
        return default
    candidate = Path(raw).expanduser()
    return candidate if candidate.is_absolute() else (ROOT_DIR / candidate).resolve()


@dataclass(frozen=True)
class Settings:
    """Resolved configuration. Access via :func:`get_settings`."""

    # Credentials. Both are None unless configured; every corpus the project
    # currently pulls is ungated, so hf_token is normally unset.
    hf_token: str | None
    kaggle_username: str | None
    kaggle_key: str | None

    # The two large directories, relocatable so ~29 GB need not sit on the
    # system drive.
    data_root: Path
    cache_root: Path

    # Raw device preference: "cuda", "cpu", or None for auto-detect. Use
    # resolve_device() rather than reading this -- it is unvalidated.
    device: str | None

    # Default checkpoint for predict and the demo server. None means "use the
    # shipping head", which those modules define; config does not own that
    # choice because it changes with every retrain.
    model_bundle: Path | None

    server_host: str
    server_port: int

    # "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", or None to default to
    # INFO. Read by aigc_detect.log.configure(); an unrecognised value falls
    # back to INFO with a warning rather than raising.
    log_level: str | None

    @property
    def has_kaggle_credentials(self) -> bool:
        return bool(self.kaggle_username and self.kaggle_key)

    @property
    def store_root(self) -> Path:
        """Where the content-addressed embedding store actually lives.

        `cache_root` holds TWO things -- the vector store under `embeddings/`
        and the path-hash memo `hashes.sqlite` -- so `cache_root` is not itself
        an `EmbeddingStore` root. Three call sites appended `"embeddings"` by
        hand and a fourth forgot, which is a silent failure rather than a loud
        one: SQLite happily CREATES a database that is not there, so the caller
        got a second, empty store beside the real one, every lookup missed, and
        the bundles it wrote carried no `bb_id` and no normalization stats --
        the revision pin stopped being load-bearing in the one artifact whose
        whole job is to carry it. Opening the store through one property rather
        than four string joins is what makes that unrepeatable.
        """
        return self.cache_root / "embeddings"

    @property
    def hash_db_path(self) -> Path:
        """The path -> (mtime, size, image id) memo, cache_root's other half."""
        return self.cache_root / "hashes.sqlite"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """The process-wide settings singleton."""
    return Settings(
        hf_token=_env("HF_TOKEN"),
        kaggle_username=_env("KAGGLE_USERNAME"),
        kaggle_key=_env("KAGGLE_KEY"),
        data_root=_env_path("AIGC_DATA_ROOT", ROOT_DIR / "data"),
        cache_root=_env_path("AIGC_CACHE_ROOT", ROOT_DIR / "data" / "cache"),
        device=_env("AIGC_DEVICE"),
        model_bundle=(Path(_env("AIGC_MODEL_BUNDLE")) if _env("AIGC_MODEL_BUNDLE") else None),
        server_host=_env("AIGC_SERVER_HOST") or "127.0.0.1",
        server_port=int(_env("AIGC_SERVER_PORT") or 8765),
        log_level=_env("AIGC_LOG_LEVEL"),
    )


def resolve_device() -> str:
    """Return the torch device string to use: "cuda" or "cpu".

    Imports torch lazily -- see the module docstring. An explicit
    ``AIGC_DEVICE=cuda`` on a machine with no visible GPU is an error rather
    than a silent fall back to CPU: this project's embedding steps take hours
    on CPU, and discovering that after the fact has cost real time.
    """
    preference = get_settings().device
    import torch

    available = torch.cuda.is_available()
    if preference is None:
        return "cuda" if available else "cpu"
    if preference == "cuda" and not available:
        raise SystemExit(
            "[settings] AIGC_DEVICE=cuda but torch.cuda.is_available() is False. "
            "Unset AIGC_DEVICE to fall back to CPU automatically, or fix the CUDA install "
            "(`uv run aigc check-env` reports what torch can see)."
        )
    return preference


def hf_token_kwargs() -> dict:
    """``{"token": ...}`` when a token is configured, else ``{}``.

    Shaped for splatting into ``huggingface_hub`` / ``datasets`` calls, so
    callers need no conditional: ``load_dataset(repo, **hf_token_kwargs())``.
    """
    token = get_settings().hf_token
    return {"token": token} if token else {}
