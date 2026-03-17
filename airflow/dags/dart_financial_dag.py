import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.insert(0, "/opt/airflow")

# 주요 상장사 corp_code 목록 (DART 기준)
CORP_CODES = [
    "00126380",  # 삼성전자
    "00164779",  # SK하이닉스
    "00401731",  # 현대자동차
    "00164742",  # LG전자
    "00138920",  # 셀트리온
    "00113494",  # 기아
    "00104235",  # POSCO홀딩스
    "00155933",  # 현대오토에버
    "00120182",  # 카카오
    "00263737",  # 네이버
]

TARGET_YEARS = [2021, 2022, 2023]
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
    from pipeline.load.postgres_loader import upsert_financials

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
