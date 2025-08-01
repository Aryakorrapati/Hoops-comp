"""
Fetch a page with all JavaScript executed, but without blocking the
FastAPI / uvicorn event-loop.  Works both inside and outside asyncio.
"""

import asyncio
from typing import Optional
from playwright.sync_api import sync_playwright
import asyncio, concurrent.futures

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
    • If we're *inside* an active asyncio loop (FastAPI request thread),
      off-load the blocking Playwright call to a new worker thread and wait
      synchronously *outside* the loop.
    • If no loop is running (CLI / local script), just call Playwright
      directly in the current thread.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We are in the uvicorn event-loop thread – create our own thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_fetch_with_playwright, url, timeout_ms)
            return future.result()          # blocks THIS thread only, loop keeps breathing
    else:
        # No running loop: simple, direct call
        return _fetch_with_playwright(url, timeout_ms)