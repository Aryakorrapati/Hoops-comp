# project/compare/play_fetch.py
from playwright.sync_api import sync_playwright

def get_html_with_js(url: str, timeout_ms: int = 30000) -> str | None:
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page    = browser.new_page()
            page.goto(url, timeout=timeout_ms)
            page.wait_for_load_state("networkidle", timeout=10000)
            html = page.content()
            browser.close()
            return html
    except Exception as e:
        print(f"[WARN] Playwright fetch failed for {url}: {e}")
        return None
