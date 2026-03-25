import hashlib

from playwright.async_api import async_playwright

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


async def scrape_naver_finance_news(stock_code: str, corp_code: str = "", max_items: int = 10) -> dict:
    """
    네이버 증권 뉴스 페이지에서 뉴스 목록을 크롤링한다.

    Args:
        stock_code: 종목코드 6자리 (예: 005930)
        corp_code:  DART 기업코드 8자리
        max_items:  최대 수집 건수

    Returns:
        {"status": "success", "stock_code": "...", "news": [...]}
    """
    # news.naver는 frameset 구조로, 뉴스 목록은 news_news.naver frame 안에 있다.
    # 부모 페이지를 로드해 쿠키/세션을 확보한 뒤 frame 내부에서 쿼리한다.
    url = f"https://finance.naver.com/item/news.naver?code={stock_code}"
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            ctx = await browser.new_context(user_agent=_USER_AGENT)
            page = await ctx.new_page()
            await page.goto(url, timeout=30000, wait_until="domcontentloaded")
            await page.wait_for_timeout(2000)

            # frameset 안의 news_news frame 탐색
            news_frame = next(
                (f for f in page.frames if "news_news" in f.url),
                None,
            )
            target = news_frame if news_frame else page

            if news_frame:
                print(f"[naver-news] {stock_code}: news_news frame 발견 ({news_frame.url[:80]})")
            else:
                print(f"[naver-news] {stock_code}: news_news frame 없음, 메인 페이지에서 시도")

            news_items = await target.evaluate(f"""
                () => {{
                    const rows = document.querySelectorAll('.type5 tbody tr');
                    const items = [];
                    for (const row of rows) {{
                        const titleEl = row.querySelector('.title a');
                        const dateEl  = row.querySelector('.date');
                        const srcEl   = row.querySelector('.info');
                        if (!titleEl) continue;
                        const title = titleEl.innerText.trim();
                        const date  = dateEl  ? dateEl.innerText.trim()  : '';
                        const src   = srcEl   ? srcEl.innerText.trim()   : '';
                        const href  = titleEl.href || '';
                        items.push({{ title, date, source: src, href, corp_code: '{corp_code}', stock_code: '{stock_code}' }});
                        if (items.length >= {max_items}) break;
                    }}
                    return items;
                }}
            """)

            # DART rcept_no 없으므로 (stock_code + title + date) MD5 14자리로 대체
            for item in news_items:
                raw = f"{item['stock_code']}_{item['title']}_{item['date']}"
                item["rcept_no"] = hashlib.md5(raw.encode()).hexdigest()[:14]

            await browser.close()
            print(f"[naver-news] {stock_code}: {len(news_items)}건 수집")
            return {"status": "success", "stock_code": stock_code, "news": news_items}

    except Exception as e:
        return {"status": "error", "stock_code": stock_code, "error_message": str(e), "news": []}


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
            context = await browser.new_context(user_agent=_USER_AGENT)
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
