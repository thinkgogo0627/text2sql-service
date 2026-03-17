import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.insert(0, "/opt/airflow")

CORP_CODES = [
    "00126380",  # 삼성전자
    "00164779",  # SK하이닉스
    "00401731",  # 현대자동차
    "00164742",  # LG전자
    "00155933",  # 현대오토에버
]

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def fetch_disclosures_task(**context):
    """DART 공시 목록 조회."""
    import os
    import requests

    api_key = os.getenv("DART_API_KEY")
    all_disclosures = []

    for corp_code in CORP_CODES:
        resp = requests.get(
            "https://opendart.fss.or.kr/api/list.json",
            params={
                "crtfc_key": api_key,
                "corp_code": corp_code,
                "bgn_de": "20230101",
                "end_de": "20231231",
                "pblntf_ty": "A",
                "page_count": 40,
            },
            timeout=30,
        )
        data = resp.json()
        if data.get("status") == "000":
            for item in data.get("list", []):
                item["corp_code"] = corp_code
                all_disclosures.append(item)

    import json
    context["ti"].xcom_push(key="disclosures", value=json.dumps(all_disclosures, ensure_ascii=False))
    print(f"Fetched {len(all_disclosures)} disclosures.")


def sentiment_analysis_task(**context):
    """Worker sLLM으로 감성분석 및 요약 수행."""
    import json
    import requests

    disclosures_json = context["ti"].xcom_pull(key="disclosures", task_ids="fetch_disclosures")
    disclosures = json.loads(disclosures_json)

    results = []
    for item in disclosures[:20]:  # 배치당 최대 20건
        try:
            # MCP 서버의 sentiment 엔드포인트 호출
            resp = requests.post(
                "http://mcp_server:8001/tools/sentiment",
                json={"text": item.get("report_nm", ""), "corp_code": item.get("corp_code")},
                timeout=30,
            )
            if resp.status_code == 200:
                sentiment = resp.json()
                results.append({
                    "corp_code":       item.get("corp_code"),
                    "rcept_no":        item.get("rcept_no"),
                    "report_nm":       item.get("report_nm"),
                    "flr_nm":          item.get("flr_nm"),
                    "rcept_dt":        item.get("rcept_dt"),
                    "rm":              item.get("rm"),
                    "sentiment_score": sentiment.get("score", 0.0),
                    "summary":         sentiment.get("summary", ""),
                    "news_content":    item.get("report_nm", ""),
                })
        except Exception as e:
            print(f"Sentiment failed for {item.get('rcept_no')}: {e}")

    context["ti"].xcom_push(key="sentiment_results", value=json.dumps(results, ensure_ascii=False))
    print(f"Sentiment analysis completed for {len(results)} records.")


def load_events_task(**context):
    """event_logs_fact에 적재."""
    import json
    import pandas as pd
    from pipeline.load.postgres_loader import upsert_event_log
    from pipeline.transform.financials import parse_date

    json_str = context["ti"].xcom_pull(key="sentiment_results", task_ids="sentiment_analysis")
    records = json.loads(json_str)

    if not records:
        print("No event logs to load.")
        return

    df = pd.DataFrame(records)
    df["rcept_dt"] = df["rcept_dt"].apply(lambda v: parse_date(str(v)) if v else None)
    upsert_event_log(df)
    print(f"Loaded {len(df)} event logs.")


with DAG(
    dag_id="sentiment_pipeline",
    default_args=default_args,
    schedule="@weekly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["dart", "sentiment"],
) as dag:

    t_fetch = PythonOperator(
        task_id="fetch_disclosures",
        python_callable=fetch_disclosures_task,
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
