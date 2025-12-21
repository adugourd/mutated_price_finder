"""HTTP utilities with retry logic and error handling."""

from __future__ import annotations

import sys
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def create_session(
    retries: int = 3,
    backoff_factor: float = 0.5,
    status_forcelist: tuple[int, ...] = (429, 500, 502, 503, 504),
    timeout: float = 30.0,
) -> requests.Session:
    """
    Create a requests session with retry logic.

    Args:
        retries: Number of retries for failed requests
        backoff_factor: Exponential backoff factor (0.5 = 0.5s, 1s, 2s...)
        status_forcelist: HTTP status codes that trigger a retry
        timeout: Default timeout for requests

    Returns:
        Configured requests.Session with retry adapters mounted
    """
    session = requests.Session()

    retry = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=list(status_forcelist),
        allowed_methods=["HEAD", "GET", "POST", "OPTIONS"],
    )

    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # Store default timeout on session for convenience
    session.timeout = timeout  # type: ignore[attr-defined]

    return session


def fetch_with_retry(
    url: str,
    session: requests.Session | None = None,
    timeout: float = 30.0,
    exit_on_error: bool = True,
    **kwargs: Any,
) -> requests.Response | None:
    """
    Fetch a URL with user-friendly error messages.

    Args:
        url: URL to fetch
        session: Optional session to use (creates one if not provided)
        timeout: Request timeout in seconds
        exit_on_error: If True, exits program on error; if False, returns None
        **kwargs: Additional arguments passed to session.get()

    Returns:
        Response object on success, None on failure (if exit_on_error=False)
    """
    if session is None:
        session = create_session()

    try:
        response = session.get(url, timeout=timeout, **kwargs)
        response.raise_for_status()
        return response

    except requests.exceptions.Timeout:
        msg = f"Error: Request to {_truncate_url(url)} timed out after {timeout}s."
        print(msg, file=sys.stderr)

    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else "unknown"
        msg = f"Error: API returned HTTP {status}. Service may be unavailable."
        print(msg, file=sys.stderr)

    except requests.exceptions.ConnectionError:
        msg = "Error: Could not connect. Check your internet connection."
        print(msg, file=sys.stderr)

    except requests.exceptions.RequestException as e:
        msg = f"Error: Request failed - {type(e).__name__}"
        print(msg, file=sys.stderr)

    if exit_on_error:
        sys.exit(1)
    return None


def _truncate_url(url: str, max_length: int = 60) -> str:
    """Truncate URL for display in error messages."""
    if len(url) <= max_length:
        return url
    return url[:max_length - 3] + "..."
