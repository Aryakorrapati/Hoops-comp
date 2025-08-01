"""
Fetch a page with all JavaScript executed, but without blocking the
FastAPI / uvicorn event-loop.  Works both inside and outside asyncio.
"""

import asyncio
from typing import Optional
from playwright.sync_api import sync_playwright

# ----- internal worker -------------------------------------------------
def _fetch_with_playwright(url: str, timeout_ms: int) -> Optional[str]:
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=["--disable-dev-shm-usage"]
            )
            page = browser.new_page()
            page.goto(url, timeout=timeout_ms)
            page.wait_for_load_state("networkidle", timeout=timeout_ms // 3)
            html = page.content()
            browser.close()
            return html
    except Exception as exc:
        print(f"[WARN] Playwright worker failed: {exc}")
        return None

# ----- public helper ---------------------------------------------------
def get_html_with_js(url: str, timeout_ms: int = 45000) -> Optional[str]:
    """
    • If called from normal synchronous code, runs Playwright in the same thread.
    • If called *inside* an asyncio event-loop (FastAPI), off-loads the
      blocking work to the default thread-pool so the loop keeps breathing.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We are inside FastAPI's event-loop – run sync Playwright in a thread.
        future = loop.run_in_executor(
            None, _fetch_with_playwright, url, timeout_ms
        )
        return future.result()
    else:
        # No running loop; just call Playwright directly.
        return _fetch_with_playwright(url, timeout_ms)
