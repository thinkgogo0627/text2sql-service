from playwright.async_api import async_playwright


async def run_playwright_scraper_tool(target_url: str, intent: str = "") -> dict:
    """
    Headless 브라우저로 URL 접속 → 본문 순수 텍스트 파싱 및 반환.

    Args:
        target_url: 크롤링 대상 URL
        intent:     크롤링 의도 (로그 용도)

    Returns:
        {"status": "success", "url": "...", "text": "...", "title": "..."}
        {"status": "error", "error_message": "..."}
    """
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            )
            page = await context.new_page()

            await page.goto(target_url, timeout=30000, wait_until="domcontentloaded")
            await page.wait_for_timeout(2000)

            title = await page.title()

            # 불필요한 요소 제거 후 텍스트 추출
            await page.evaluate("""
                () => {
                    ['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe']
                        .forEach(tag => {
                            document.querySelectorAll(tag).forEach(el => el.remove());
                        });
                }
            """)

            text = await page.inner_text("body")
            # 공백 정리
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            clean_text = "\n".join(lines)

            await browser.close()

            return {
                "status": "success",
                "url": target_url,
                "title": title,
                "text": clean_text[:10000],  # 최대 10,000자
                "intent": intent,
            }

    except Exception as e:
        return {
            "status": "error",
            "url": target_url,
            "error_message": str(e),
        }
