"""WSGI entrypoint for Render (run from repository root)."""

from webapp.memory_utils import configure_torch

configure_torch()

from webapp.app import app  # noqa: E402
