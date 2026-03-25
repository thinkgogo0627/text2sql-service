import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.insert(0, "/opt/airflow")

# dart_financial_dag.py의 CORP_CODES와 동일하게 유지
# stock_code가 있는 상장사만 포함 (company_dim 등록 전제)
CORP_CODES = [
    "00126380",  # 삼성전자        (005930)
    "00164779",  # SK하이닉스      (000660)
    "00126371",  # 삼성전기        (009150)
    "00266961",  # NAVER           (035420)
    "00258801",  # 카카오          (035720)
    "01133217",  # 카카오뱅크      (323410)
    "00760971",  # 크래프톤        (259960)
    "00261464",  # 엔씨소프트      (036570)
    "00835456",  # 넷마블          (251270)
    "00164742",  # 현대자동차      (005380)
    "00106641",  # 기아            (000270)
    "00164788",  # 현대모비스      (012330)
    "00164792",  # 현대제철        (004020)
    "00699241",  # 현대글로비스    (086280)
    "00401731",  # LG전자          (066570)
    "00105873",  # LG디스플레이    (034220)
    "00126362",  # 삼성SDI         (006400)
    "00356361",  # LG화학          (051910)
    "01515323",  # LG에너지솔루션  (373220)
    "00155319",  # POSCO홀딩스     (005490)
    "00631518",  # SK이노베이션    (096770)
    "00165413",  # 롯데케미칼      (011170)
    "00179909",  # 한화솔루션      (009830)
    "00946205",  # 에코프로비엠    (247540)
    "00128958",  # S-Oil           (010950)
    "00413046",  # 셀트리온        (068270)
    "00877059",  # 삼성바이오로직스 (207940)
    "00420081",  # 한미약품        (128940)
    "00100699",  # 유한양행        (000100)
    "00159023",  # SK텔레콤        (017670)
    "00190321",  # KT              (030200)
    "00215243",  # LG유플러스      (032640)
    "00688996",  # KB금융          (105560)
    "00382199",  # 신한지주        (055550)
    "00547583",  # 하나금융지주    (086790)
    "01350869",  # 우리금융지주    (316140)
    "00126256",  # 삼성생명        (032830)
    "00114814",  # 삼성화재        (000810)
    "00164049",  # 메리츠금융지주  (138040)
    "00159055",  # 미래에셋증권    (006800)
    "00164478",  # 현대건설        (000720)
    "00159616",  # 두산에너빌리티  (034020)
    "00104032",  # 한화에어로스페이스 (012450)
    "00149655",  # 삼성물산        (028260)
    "00693796",  # 이마트          (139480)
    "00388255",  # 아모레퍼시픽    (090430)
    "00356680",  # LG생활건강      (051900)
    "00159193",  # 한국전력공사    (015760)
    "00159916",  # 대한항공        (003490)
    # 
    "00302926",  # 현대로템
    "00362441",  # 현대오토에버
    "00111704",  # 한화오션
    "00113058",  # 한화생명
    "00159795",  # 한국카본
    "00105961",  # LG이노텍
    "00139834",  # LG씨엔에스
    "00842619",  # 리가켐바이오
    "01786958", # 큐리오시스
    "00098826",
    "00108436",
    "00149868",
    "00154953",
    '00164308',
    '00591572'
]

_DATA_DIR = "/opt/airflow/data/raw"


def _news_path(run_id: str) -> str:
    return f"{_DATA_DIR}/naver_news_{run_id}.json"


def _sentiment_path(run_id: str) -> str:
    return f"{_DATA_DIR}/sentiment_results_{run_id}.json"

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def fetch_naver_news_task(**context):
    """
    company_dim에서 stock_code 조회 후
    네이버 증권 뉴스 페이지를 Playwright로 크롤링한다.
    """
    import json
    import requests
    from sqlalchemy import text
    from pipeline.load.postgres_loader import get_engine

    news_path = _news_path(context["run_id"])

    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT corp_code, stock_code FROM company_dim WHERE corp_code = ANY(:codes)"),
            {"codes": CORP_CODES},
        )
        corp_stock_map = {r.corp_code: r.stock_code for r in rows if r.stock_code}

    if not corp_stock_map:
        raise ValueError("company_dim에서 stock_code를 찾을 수 없습니다. seed_company_dim을 먼저 실행하세요.")

    all_news = []
    for corp_code, stock_code in corp_stock_map.items():
        try:
            resp = requests.post(
                "http://mcp_server:8001/tools/naver-news",
                json={"stock_code": stock_code, "corp_code": corp_code, "max_items": 15},
                timeout=60,
            )
            if resp.status_code == 200:
                result = resp.json()
                news_list = result.get("news", [])
                all_news.extend(news_list)
                print(f"Fetched {len(news_list)} news: {corp_code} ({stock_code})")
            else:
                print(f"naver-news API 오류: {resp.status_code} | corp={corp_code}")
        except Exception as e:
            print(f"뉴스 크롤링 실패: {corp_code} ({stock_code}): {e}")

    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(news_path, "w", encoding="utf-8") as f:
        json.dump(all_news, f, ensure_ascii=False)
    print(f"총 {len(all_news)}개 뉴스 → {news_path}")


def sentiment_analysis_task(**context):
    """
    크롤링한 뉴스 제목을 Llama 8B로 감성분석 및 3줄 요약.
    """
    import json
    import requests

    run_id = context["run_id"]
    news_path = _news_path(run_id)
    sentiment_path = _sentiment_path(run_id)

    if not os.path.exists(news_path):
        raise FileNotFoundError(f"뉴스 데이터 파일 없음: {news_path}")

    with open(news_path, encoding="utf-8") as f:
        news_list = json.load(f)

    if not news_list:
        print("분석할 뉴스가 없습니다.")
        with open(sentiment_path, "w") as f:
            json.dump([], f)
        os.remove(news_path)
        return

    results = []
    for item in news_list:
        try:
            resp = requests.post(
                "http://mcp_server:8001/tools/sentiment",
                json={"text": item.get("title", ""), "corp_code": item.get("corp_code", "")},
                timeout=30,
            )
            if resp.status_code == 200:
                sentiment = resp.json()
                results.append({
                    "corp_code":       item.get("corp_code"),
                    "rcept_no":        item.get("rcept_no"),
                    "report_nm":       item.get("title", ""),
                    "flr_nm":          item.get("source", ""),
                    "rcept_dt":        item.get("date", ""),
                    "rm":              None,
                    "news_content":    item.get("title", ""),
                    "sentiment_score": sentiment.get("sentiment_score", 0.0),
                    "summary":         sentiment.get("summary", ""),
                })
            else:
                print(f"Sentiment API 오류: {resp.status_code} | rcept_no={item.get('rcept_no')}")
        except Exception as e:
            print(f"Sentiment 실패: {item.get('rcept_no')}: {e}")

    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(sentiment_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    print(f"감성분석 완료: {len(results)}/{len(news_list)}건 → {sentiment_path}")

    os.remove(news_path)
    print(f"임시 파일 삭제: {news_path}")


def load_events_task(**context):
    """event_logs_fact에 감성분석 결과 적재."""
    import json
    import pandas as pd
    from pipeline.load.postgres_loader import upsert_event_log
    from pipeline.transform.financials import parse_date

    sentiment_path = _sentiment_path(context["run_id"])

    if not os.path.exists(sentiment_path):
        # 재시도 시 이미 성공적으로 처리되어 파일이 삭제된 경우 정상 종료
        if context.get("ti").try_number > 1:
            print(f"재시도({context['ti'].try_number})에서 파일 없음 — 이전 시도에서 이미 처리된 것으로 간주합니다.")
            return
        raise FileNotFoundError(f"감성분석 결과 파일 없음: {sentiment_path}")

    with open(sentiment_path, encoding="utf-8") as f:
        records = json.load(f)

    if not records:
        print("적재할 이벤트 로그가 없습니다.")
        os.remove(sentiment_path)
        return

    df = pd.DataFrame(records)
    df["rcept_dt"] = df["rcept_dt"].apply(lambda v: parse_date(str(v).replace(".", "").strip()) if v else None)
    upsert_event_log(df)
    print(f"event_logs_fact에 {len(df)}건 적재.")

    os.remove(sentiment_path)
    print(f"임시 파일 삭제: {sentiment_path}")


with DAG(
    dag_id="sentiment_pipeline",
    default_args=default_args,
    schedule=timedelta(days=3),
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["naver", "sentiment"],
) as dag:

    t_fetch = PythonOperator(
        task_id="fetch_naver_news",
        python_callable=fetch_naver_news_task,
    )

    t_sentiment = PythonOperator(
        task_id="sentiment_analysis",
        python_callable=sentiment_analysis_task,
    )

    t_load = PythonOperator(
        task_id="load_events",
        python_callable=load_events_task,
    )

    t_fetch >> t_sentiment >> t_load
