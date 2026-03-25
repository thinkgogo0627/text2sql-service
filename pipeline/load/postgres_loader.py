import os
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

DDL_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS company_dim (
        corp_code     VARCHAR(8)   PRIMARY KEY,
        corp_name     VARCHAR(100) NOT NULL,
        corp_name_eng VARCHAR(200),
        stock_name    VARCHAR(100),
        stock_code    VARCHAR(6),
        ceo_nm        VARCHAR(200),
        corp_cls      CHAR(1),
        jurir_no      VARCHAR(13),
        bizr_no       VARCHAR(10),
        adres         TEXT,
        hm_url        VARCHAR(200),
        phn_no        VARCHAR(20),
        induty_code   VARCHAR(6),
        est_dt        DATE,
        acc_mt        CHAR(2),
        updated_at    TIMESTAMP DEFAULT NOW()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS financial_fact (
        id            BIGSERIAL    PRIMARY KEY,
        corp_code     VARCHAR(8)   NOT NULL REFERENCES company_dim(corp_code),
        rcept_no      VARCHAR(14),
        reprt_code    VARCHAR(5),
        bsns_year     SMALLINT     NOT NULL,
        fs_div        CHAR(3)      NOT NULL,
        sj_div        VARCHAR(3)   NOT NULL,
        sj_nm         VARCHAR(50),
        account_id    VARCHAR(100),
        account_nm    VARCHAR(200) NOT NULL,
        account_detail TEXT,
        amount        BIGINT,
        currency      CHAR(3)      DEFAULT 'KRW',
        ord           SMALLINT,
        created_at    TIMESTAMP    DEFAULT NOW(),
        UNIQUE (corp_code, bsns_year, fs_div, sj_div, account_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS event_logs_fact (
        id              BIGSERIAL    PRIMARY KEY,
        corp_code       VARCHAR(8)   NOT NULL REFERENCES company_dim(corp_code),
        rcept_no        VARCHAR(14)  UNIQUE,
        report_nm       VARCHAR(500),
        flr_nm          VARCHAR(200),
        rcept_dt        DATE,
        rm              CHAR(1),
        news_content    TEXT,
        sentiment_score FLOAT,
        summary         TEXT,
        created_at      TIMESTAMP    DEFAULT NOW()
    )
    """,
]


def get_engine():
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "text2sql_db")
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
    return create_engine(url)


def create_tables():
    engine = get_engine()
    with engine.begin() as conn:
        for ddl in DDL_STATEMENTS:
            conn.execute(text(ddl))
    print("Tables created successfully.")


def upsert_company(df: pd.DataFrame):
    """company_dim에 기업 정보를 upsert."""
    engine = get_engine()
    upsert_sql = text("""
        INSERT INTO company_dim (
            corp_code, corp_name, corp_name_eng, stock_name, stock_code,
            ceo_nm, corp_cls, jurir_no, bizr_no, adres, hm_url, phn_no,
            induty_code, est_dt, acc_mt, updated_at
        ) VALUES (
            :corp_code, :corp_name, :corp_name_eng, :stock_name, :stock_code,
            :ceo_nm, :corp_cls, :jurir_no, :bizr_no, :adres, :hm_url, :phn_no,
            :induty_code, :est_dt, :acc_mt, NOW()
        )
        ON CONFLICT (corp_code) DO UPDATE SET
            corp_name     = EXCLUDED.corp_name,
            corp_name_eng = EXCLUDED.corp_name_eng,
            stock_name    = EXCLUDED.stock_name,
            stock_code    = EXCLUDED.stock_code,
            ceo_nm        = EXCLUDED.ceo_nm,
            updated_at    = NOW()
    """)
    with engine.begin() as conn:
        for _, row in df.iterrows():
            conn.execute(upsert_sql, row.to_dict())
    print(f"Upserted {len(df)} company records.")


def upsert_financials(df: pd.DataFrame):
    """financial_fact에 재무 데이터를 upsert."""
    engine = get_engine()
    upsert_sql = text("""
        INSERT INTO financial_fact (
            corp_code, rcept_no, reprt_code, bsns_year, fs_div,
            sj_div, sj_nm, account_id, account_nm, account_detail,
            amount, currency, ord
        ) VALUES (
            :corp_code, :rcept_no, :reprt_code, :bsns_year, :fs_div,
            :sj_div, :sj_nm, :account_id, :account_nm, :account_detail,
            :amount, :currency, :ord
        )
        ON CONFLICT (corp_code, bsns_year, fs_div, sj_div, account_id) DO UPDATE SET
            amount     = EXCLUDED.amount,
            rcept_no   = EXCLUDED.rcept_no,
            created_at = NOW()
    """)
    records = df.where(pd.notnull(df), None).to_dict(orient="records")
    # corp_code 8자리 보정 (pandas read_json 타입 추론으로 leading zero 소실 방어)
    for r in records:
        if r.get("corp_code") is not None:
            r["corp_code"] = str(r["corp_code"]).zfill(8)
    success = 0
    failed = 0
    for record in records:
        try:
            with engine.begin() as conn:
                conn.execute(upsert_sql, record)
            success += 1
        except SQLAlchemyError as e:
            print(f"Row upsert failed: {e} | corp_code={record.get('corp_code')} bsns_year={record.get('bsns_year')}")
            failed += 1
    print(f"Upserted {success}/{len(records)} financial records. Failed: {failed}")
    if failed > 0:
        raise RuntimeError(f"{failed}개 row 적재 실패. 로그를 확인하세요.")


def upsert_event_log(df: pd.DataFrame):
    """event_logs_fact에 공시/뉴스 로그를 upsert."""
    engine = get_engine()
    upsert_sql = text("""
        INSERT INTO event_logs_fact (
            corp_code, rcept_no, report_nm, flr_nm, rcept_dt,
            rm, news_content, sentiment_score, summary
        ) VALUES (
            :corp_code, :rcept_no, :report_nm, :flr_nm, :rcept_dt,
            :rm, :news_content, :sentiment_score, :summary
        )
        ON CONFLICT (rcept_no) DO UPDATE SET
            news_content    = EXCLUDED.news_content,
            sentiment_score = EXCLUDED.sentiment_score,
            summary         = EXCLUDED.summary
    """)
    records = df.where(pd.notnull(df), None).to_dict(orient="records")
    for r in records:
        if r.get("corp_code") is not None:
            r["corp_code"] = str(r["corp_code"]).zfill(8)
    success = 0
    failed = 0
    for record in records:
        try:
            with engine.begin() as conn:
                conn.execute(upsert_sql, record)
            success += 1
        except SQLAlchemyError as e:
            print(f"Event log upsert failed: {e} | corp_code={record.get('corp_code')} rcept_no={record.get('rcept_no')}")
            failed += 1
    print(f"Upserted {success}/{len(records)} event log records. Failed: {failed}")
    if failed > 0:
        raise RuntimeError(f"event_log {failed}개 row 적재 실패. 위 로그를 확인하세요.")


if __name__ == "__main__":
    create_tables()
