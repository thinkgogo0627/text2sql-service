import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.insert(0, "/opt/airflow")

# 주요 상장사 corp_code 목록 (DART 기준, stock_code로 검증)
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
    "00302926",  # 현대로템
    "00362441",  # 현대오토에버
    "00111704",  # 한화오션
    "00113058",  # 한화생명
    "00159795",  # 한국카본
    "00105961",  # LG이노텍
    "00139834",  # LG씨엔에스
    "00842619",  # 리가켐바이오
    "01786958",  # 큐리오시스
    "00098826",
    "00108436",
    "00149868",
    "00154953",
    '00164308',
    '00591572'
]

TARGET_YEARS = list(range(2020, 2027))  # 2020 ~ 2026
REPRT_CODE = "11011"  # 사업보고서

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def seed_company_dim_task(**context):
    """DART 기업개황 API로 company_dim을 먼저 채운다."""
    import requests
    import pandas as pd
    from pipeline.load.postgres_loader import create_tables, upsert_company

    create_tables()

    api_key = os.getenv("DART_API_KEY")
    url = "https://opendart.fss.or.kr/api/company.json"

    records = []
    for corp_code in CORP_CODES:
        try:
            resp = requests.get(url, params={"crtfc_key": api_key, "corp_code": corp_code}, timeout=10)
            resp.raise_for_status()
            d = resp.json()
            if d.get("status") != "000":
                print(f"company info 오류: {d.get('message')} (corp={corp_code})")
                continue
            records.append({
                "corp_code":     d.get("corp_code", "").strip(),
                "corp_name":     d.get("corp_name", "").strip(),
                "corp_name_eng": d.get("corp_name_eng", "").strip() or None,
                "stock_name":    d.get("stock_name", "").strip() or None,
                "stock_code":    d.get("stock_code", "").strip() or None,
                "ceo_nm":        d.get("ceo_nm", "").strip() or None,
                "corp_cls":      d.get("corp_cls", "").strip() or None,
                "jurir_no":      d.get("jurir_no", "").strip() or None,
                "bizr_no":       d.get("bizr_no", "").strip() or None,
                "adres":         d.get("adres", "").strip() or None,
                "hm_url":        d.get("hm_url", "").strip() or None,
                "phn_no":        d.get("phn_no", "").strip() or None,
                "induty_code":   d.get("induty_code", "").strip() or None,
                "est_dt":        d.get("est_dt", "").strip() or None,
                "acc_mt":        d.get("acc_mt", "").strip() or None,
            })
            print(f"Fetched company info: {d.get('corp_name')} ({corp_code})")
        except Exception as e:
            print(f"company info 실패: {corp_code}: {e}")

    if records:
        df = pd.DataFrame(records)
        upsert_company(df)
        print(f"company_dim에 {len(df)}개 기업 정보 적재 완료.")
    else:
        raise ValueError("company_dim 적재 실패: 가져온 기업 정보가 없습니다.")


RAW_DATA_PATH = "/opt/airflow/data/raw/raw_data.json"
TRANSFORMED_DATA_PATH = "/opt/airflow/data/raw/transformed_data.json"


def extract_task(**context):
    import json
    from pipeline.extract.dart_api import extract_dart_api

    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    results = {}
    for corp_code in CORP_CODES:
        for year in TARGET_YEARS:
            try:
                df = extract_dart_api(corp_code, year, REPRT_CODE)
                if not df.empty:
                    results[f"{corp_code}_{year}"] = df.to_json(orient="records")
                    print(f"Extracted {len(df)} rows for {corp_code} {year}")
                else:
                    print(f"No data for {corp_code} {year} (skipped)")
            except Exception as e:
                print(f"Extraction failed for {corp_code} {year}: {e}")

    with open(RAW_DATA_PATH, "w") as f:
        json.dump(results, f)
    print(f"Extract 완료: {len(results)}건 → {RAW_DATA_PATH}")


def transform_task(**context):
    import json
    import pandas as pd
    from pipeline.transform.financials import transform_and_load_financials

    if not os.path.exists(RAW_DATA_PATH):
        print(f"raw_data 파일을 찾을 수 없습니다: {RAW_DATA_PATH}")
        os.makedirs(os.path.dirname(TRANSFORMED_DATA_PATH), exist_ok=True)
        with open(TRANSFORMED_DATA_PATH, "w") as f:
            json.dump([], f)
        return

    with open(RAW_DATA_PATH) as f:
        raw_data = json.load(f)

    if not raw_data:
        print("raw_data가 비어있습니다.")
        os.makedirs(os.path.dirname(TRANSFORMED_DATA_PATH), exist_ok=True)
        with open(TRANSFORMED_DATA_PATH, "w") as f:
            json.dump([], f)
        os.remove(RAW_DATA_PATH)
        return

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

    os.makedirs(os.path.dirname(TRANSFORMED_DATA_PATH), exist_ok=True)
    if all_transformed:
        df_all = pd.concat(all_transformed, ignore_index=True)
        df_all.to_json(TRANSFORMED_DATA_PATH, orient="records", force_ascii=False)
        print(f"Transform 완료: 총 {len(df_all)}행 → {TRANSFORMED_DATA_PATH}")
    else:
        with open(TRANSFORMED_DATA_PATH, "w") as f:
            json.dump([], f)
        print("Transform 결과 없음.")

    os.remove(RAW_DATA_PATH)
    print(f"임시 raw 파일 삭제: {RAW_DATA_PATH}")


def load_task(**context):
    import pandas as pd
    from pipeline.load.postgres_loader import upsert_financials

    if not os.path.exists(TRANSFORMED_DATA_PATH):
        raise FileNotFoundError(
            f"변환 데이터 파일을 찾을 수 없습니다: {TRANSFORMED_DATA_PATH}. "
            "transform_task가 성공적으로 실행됐는지 확인하세요."
        )

    df = pd.read_json(TRANSFORMED_DATA_PATH, orient="records")

    if df.empty:
        print("No data to load.")
        os.remove(TRANSFORMED_DATA_PATH)
        return

    upsert_financials(df)
    print(f"financial_fact에 {len(df)}행 적재 완료.")
    os.remove(TRANSFORMED_DATA_PATH)
    print(f"임시 파일 삭제 완료: {TRANSFORMED_DATA_PATH}")


with DAG(
    dag_id="dart_financial_pipeline",
    default_args=default_args,
    schedule="@quarterly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["dart", "financials"],
) as dag:

    t_seed = PythonOperator(
        task_id="seed_company_dim",
        python_callable=seed_company_dim_task,
    )

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

    t_seed >> t_extract >> t_transform >> t_load
