"""Keep the Streamlit Community Cloud app awake and its caches warm.

Community Cloud puts an app to sleep after a quiet spell, and a sleeping app
shows a "wake it up" page instead of the demo. A headless browser visits the
heaviest pages on a schedule, presses the wake button if it appears, and waits
for each page to finish rendering so the computed results are cached.
"""

import os
import sys

from playwright.sync_api import TimeoutError as PWTimeout
from playwright.sync_api import sync_playwright

BASE = os.environ.get("APP_URL", "https://richardhenderson-lab.streamlit.app").rstrip("/")
PAGES = ["", "pricing-copilot?stage=model", "pricing-copilot?stage=prove", "market-intelligence", "lending"]
WAKE_TEXT = ["Yes, get this app back up", "get this app back up"]


def visit(page, url: str) -> bool:
    page.goto(url, wait_until="domcontentloaded", timeout=120_000)
    for text in WAKE_TEXT:
        button = page.get_by_role("button", name=text)
        if button.count():
            print(f"  asleep; waking via '{text}'")
            button.first.click()
            page.wait_for_timeout(5_000)
            break
    # The app renders inside an iframe on Community Cloud; look in every frame.
    for _ in range(120):
        for frame in page.frames:
            try:
                if frame.locator('[data-testid="stAppViewContainer"]').count() and \
                        not frame.locator('[data-testid="stStatusWidget"]').count():
                    page.wait_for_timeout(2_000)
                    return True
            except Exception:
                pass
        page.wait_for_timeout(1_000)
    return False


def main() -> int:
    ok = True
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        for path in PAGES:
            url = f"{BASE}/{path}"
            try:
                rendered = visit(page, url)
            except PWTimeout:
                rendered = False
            print(f"{'ok  ' if rendered else 'FAIL'} {url}")
            ok &= rendered
        browser.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
