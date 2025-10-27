"""Runtime configuration for the Life demo.

Structured as plain Python data to keep things simple while still allowing
an easy port to JSON/TOML later. The top-level dictionary holds presets, and
`ACTIVE_PROFILE` selects which one is used at runtime.
"""

from __future__ import annotations

ACTIVE_PROFILE = "smooth_demo"


PROFILES: dict[str, dict[str, object]] = {
    "classic_demo": {
        "size": 20,
        "steps": 100,
        "wrap": True,
        "mode": "classic",
    },
    "smooth_demo": {
        "size": 20,
        "steps": 100,
        "wrap": True,
        "mode": "smooth",
        "smooth": {
            "alpha": 8.0,
            "sigma": 0.20,
            "hard": False,
        },
    },
}


def get_active_config() -> dict[str, object]:
    """Return the configuration dictionary for the active profile."""

    cfg = PROFILES.get(ACTIVE_PROFILE)
    if cfg is None:
        available = ", ".join(sorted(PROFILES))
        msg = f"Unknown profile '{ACTIVE_PROFILE}'. Available: {available}"
        raise KeyError(msg)
    return cfg
