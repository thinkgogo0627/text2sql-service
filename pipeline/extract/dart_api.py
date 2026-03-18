import os
import json
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

DART_API_URL = "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json"
DATA_LAKE_PATH = os.getenv("DATA_LAKE_PATH", "./data/raw")


def extract_dart_api(corp_code: str, year: int, reprt_code: str = "11011") -> pd.DataFrame:
    """
    OpenDART API에서 단일기업 전체 재무제표를 조회한다.

    Args:
        corp_code:   DART 기업 고유코드 (8자리)
        year:        사업연도 (예: 2023)
        reprt_code:  보고서 코드 (11011=사업보고서, 11012=반기, 11013=1분기, 11014=3분기)

    Returns:
        DataFrame (raw API 응답 list)
    """
    api_key = os.getenv("DART_API_KEY")
    if not api_key:
        raise ValueError("DART_API_KEY 환경변수가 설정되지 않았습니다.")

    params = {
        "crtfc_key": api_key,
        "corp_code": corp_code,
        "bsns_year": str(year),
        "reprt_code": reprt_code,
        "fs_div": "CFS",
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.get(DART_API_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            if data.get("status") != "000":
                print(f"DART API 오류: {data.get('message')} (corp={corp_code}, year={year})")
                return pd.DataFrame()

            records = data.get("list", [])
            df = pd.DataFrame(records)
            if not df.empty:
                df["fs_div"] = params["fs_div"]
            try:
                _save_raw(df, corp_code, year, reprt_code, data)
            except Exception as save_err:
                print(f"Raw 저장 실패 (무시됨): {save_err}")
            return df

        except requests.exceptions.RequestException as e:
            wait = 2 ** attempt
            print(f"API 호출 실패 (시도 {attempt + 1}/{max_retries}): {e}. {wait}초 후 재시도...")
            if attempt < max_retries - 1:
                time.sleep(wait)
            else:
                raise

    return pd.DataFrame()


def _save_raw(df: pd.DataFrame, corp_code: str, year: int, reprt_code: str, raw_json: dict):
    """원본 JSON을 data/raw/financials/{year}/{corp_code}_{reprt_code}.json 에 저장."""
    save_dir = Path(DATA_LAKE_PATH) / "financials" / str(year)
    save_dir.mkdir(parents=True, exist_ok=True)
    file_path = save_dir / f"{corp_code}_{reprt_code}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(raw_json, f, ensure_ascii=False, indent=2)
    print(f"Raw JSON saved: {file_path}")


def fetch_corp_list() -> pd.DataFrame:
    """
    DART 기업 목록 전체를 다운로드한다 (ZIP → XML 파싱).
    Returns DataFrame with corp_code, corp_name, stock_code, ...
    """
    import zipfile
    import io
    import xml.etree.ElementTree as ET

    api_key = os.getenv("DART_API_KEY")
    url = "https://opendart.fss.or.kr/api/corpCode.xml"
    resp = requests.get(url, params={"crtfc_key": api_key}, timeout=60)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        xml_filename = [n for n in zf.namelist() if n.endswith(".xml")][0]
        with zf.open(xml_filename) as xml_file:
            tree = ET.parse(xml_file)

    root = tree.getroot()
    records = []
    for item in root.findall("list"):
        records.append({
            "corp_code":  item.findtext("corp_code", "").strip(),
            "corp_name":  item.findtext("corp_name", "").strip(),
            "stock_code": item.findtext("stock_code", "").strip() or None,
            "modify_date": item.findtext("modify_date", "").strip(),
        })
    return pd.DataFrame(records)
