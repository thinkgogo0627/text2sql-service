import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.insert(0, "/opt/airflow")

# 주요 상장사 corp_code 목록 (DART 기준, stock_code로 검증)
CORP_CODES = [
    # IT / 반도체
    "00126380",  # 삼성전자     (005930)
    "00164779",  # SK하이닉스   (000660)
    "00266961",  # NAVER        (035420)
    "00258801",  # 카카오       (035720)
    "00760971",  # 크래프톤     (259960)
    "01133217",  # 카카오뱅크   (323410)
    # 자동차
    "00164742",  # 현대자동차   (005380)
    "00106641",  # 기아         (000270)
    "00164788",  # 현대모비스   (012330)
    # 전자 / 디스플레이
    "00401731",  # LG전자       (066570)
    "00105873",  # LG디스플레이 (034220)
    "00126362",  # 삼성SDI      (006400)
    # 화학 / 에너지
    "00356361",  # LG화학       (051910)
    "01515323",  # LG에너지솔루션 (373220)
    "00155319",  # POSCO홀딩스  (005490)
    "00631518",  # SK이노베이션 (096770)
    "00165413",  # 롯데케미칼   (011170)
    # 바이오 / 헬스케어
    "00413046",  # 셀트리온     (068270)
    "00877059",  # 삼성바이오로직스 (207940)
    # 통신
    "00159023",  # SK텔레콤     (017670)
    "00190321",  # KT           (030200)
    # 금융
    "00688996",  # KB금융       (105560)
    "00382199",  # 신한지주     (055550)
    "00547583",  # 하나금융지주 (086790)
    "01350869",  # 우리금융지주 (316140)
    "00126256",  # 삼성생명     (032830)
    # 건설 / 중공업
    "00164478",  # 현대건설     (000720)
    "00159616",  # 두산에너빌리티 (034020)
    # 유통 / 기타
    "00149655",  # 삼성물산     (028260)
    "00159193",  # 한국전력공사 (015760)
]

TARGET_YEARS = list(range(2020, 2027))  # 2020 ~ 2026
REPRT_CODE = "11011"  # 사업보고서

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def extract_task(**context):
    from pipeline.extract.dart_api import extract_dart_api
    results = {}
    for corp_code in CORP_CODES:
        for year in TARGET_YEARS:
            try:
                df = extract_dart_api(corp_code, year, REPRT_CODE)
                results[f"{corp_code}_{year}"] = df.to_json(orient="records")
                print(f"Extracted {len(df)} rows for {corp_code} {year}")
            except Exception as e:
                print(f"Extraction failed for {corp_code} {year}: {e}")
    context["ti"].xcom_push(key="raw_data", value=results)


def transform_task(**context):
    import json
    import pandas as pd
    from pipeline.transform.financials import transform_and_load_financials

    raw_data = context["ti"].xcom_pull(key="raw_data", task_ids="extract_dart_api")
    all_transformed = []

    for key, json_str in raw_data.items():
        try:
            df_raw = pd.read_json(json_str, orient="records")
            df_long = transform_and_load_financials(df_raw)
            if not df_long.empty:
                all_transformed.append(df_long)
                print(f"Transformed {len(df_long)} rows for {key}")
        except Exception as e:
            print(f"Transform failed for {key}: {e}")

    if all_transformed:
        df_all = pd.concat(all_transformed, ignore_index=True)
        context["ti"].xcom_push(key="transformed_data", value=df_all.to_json(orient="records"))
    else:
        context["ti"].xcom_push(key="transformed_data", value="[]")


def load_task(**context):
    import pandas as pd
    from pipeline.load.postgres_loader import create_tables, upsert_financials

    create_tables()

    json_str = context["ti"].xcom_pull(key="transformed_data", task_ids="transform_and_load_financials")
    df = pd.read_json(json_str, orient="records")

    if df.empty:
        print("No data to load.")
        return

    upsert_financials(df)
    print(f"Loaded {len(df)} rows into financial_fact.")


with DAG(
    dag_id="dart_financial_pipeline",
    default_args=default_args,
    schedule="@quarterly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["dart", "financials"],
) as dag:

    t_extract = PythonOperator(
        task_id="extract_dart_api",
        python_callable=extract_task,
    )

    t_transform = PythonOperator(
        task_id="transform_and_load_financials",
        python_callable=transform_task,
    )

    t_load = PythonOperator(
        task_id="upsert_financials",
        python_callable=load_task,
    )

    t_extract >> t_transform >> t_load
