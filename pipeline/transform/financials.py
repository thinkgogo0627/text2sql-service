import re
import pandas as pd
from datetime import datetime


def parse_year(nm_value: str, fallback: int) -> int:
    """
    API의 thstrm_nm / frmtrm_nm 값에서 연도를 파싱한다.
    예) "제 55 기" → fallback 사용, "2023년" → 2023
    """
    if nm_value:
        match = re.search(r"(\d{4})", nm_value)
        if match:
            return int(match.group(1))
    return fallback


def transform_and_load_financials(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    DART API 원본 Wide 포맷 → Long 포맷으로 변환한다.

    Wide → Long Unpivot 처리:
      API 원본 1 row → 최대 3 rows (당기/전기/전전기)

    Returns:
        financial_fact 테이블에 적재 가능한 Long 포맷 DataFrame
    """
    if df_raw.empty:
        return pd.DataFrame()

    required_cols = ["corp_code", "bsns_year", "fs_div", "sj_div", "account_nm"]
    for col in required_cols:
        if col not in df_raw.columns:
            raise ValueError(f"필수 컬럼 누락: {col}")

    records = []

    for _, row in df_raw.iterrows():
        base = {
            "corp_code":      str(row.get("corp_code", "")).strip(),
            "rcept_no":       str(row.get("rcept_no", "")).strip() or None,
            "reprt_code":     str(row.get("reprt_code", "")).strip() or None,
            "fs_div":         str(row.get("fs_div", "")).strip(),
            "sj_div":         str(row.get("sj_div", "")).strip(),
            "sj_nm":          str(row.get("sj_nm", "")).strip() or None,
            "account_id":     str(row.get("account_id", "")).strip() or None,
            "account_nm":     str(row.get("account_nm", "")).strip(),
            "account_detail": str(row.get("account_detail", "")).strip() or None,
            "currency":       "KRW",
            "ord":            _safe_int(row.get("ord")),
        }

        base_year = _safe_int(row.get("bsns_year")) or 0

        # 당기 / 전기 / 전전기 언피벗
        for nm_col, amt_col, fallback in [
            ("thstrm_nm",    "thstrm_amount",    base_year),
            ("frmtrm_nm",    "frmtrm_amount",    base_year - 1),
            ("bfefrmtrm_nm", "bfefrmtrm_amount", base_year - 2),
        ]:
            amt = pd.to_numeric(
                str(row.get(amt_col, "")).replace(",", ""),
                errors="coerce"
            )
            if pd.isna(amt):
                continue
            year_val = parse_year(str(row.get(nm_col, "")), fallback)
            records.append({
                **base,
                "bsns_year": year_val,
                "amount": int(amt),
            })

    if not records:
        return pd.DataFrame()

    df_long = pd.DataFrame(records)

    # 중복 제거 (corp_code, bsns_year, fs_div, sj_div, account_id 기준)
    subset = ["corp_code", "bsns_year", "fs_div", "sj_div", "account_id"]
    df_long = df_long.drop_duplicates(subset=subset, keep="last")

    return df_long


def _safe_int(val) -> int | None:
    try:
        return int(str(val).replace(",", "").strip())
    except (ValueError, TypeError):
        return None


def parse_date(val: str) -> datetime | None:
    """'20231211' → datetime 변환."""
    if not val or not str(val).strip():
        return None
    try:
        return datetime.strptime(str(val).strip(), "%Y%m%d")
    except ValueError:
        return None
