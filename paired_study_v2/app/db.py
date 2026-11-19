"""
app/db.py
=========
PostgreSQL schema management and low-level data access helpers.

Schema is applied idempotently via ALTER TABLE … ADD COLUMN IF NOT EXISTS so
the same migration can run against both a fresh container and an existing
Docker volume without errors.

Public API:
  config_from_env()         -> DbConfig
  connect(cfg)              -> psycopg.Connection
  init_schema(conn)         -> None
  insert_record(conn, ...)  -> (id, db_ms)
  fetch_record(conn, ...)   -> (DbEncryptedRow, db_ms)
  update_wrap_fields(conn, ...) -> db_ms
  soft_delete_record(conn, ...) -> db_ms
  audit_event(conn, ...)    -> None
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Any
import os
import time

import psycopg


# -----------------------------
# Configuration / Connection
# -----------------------------

@dataclass(frozen=True)
class DbConfig:
    host: str
    port: int
    dbname: str
    user: str
    password: str


def config_from_env(prefix: str = "PG") -> DbConfig:
    """Reads standard Postgres env vars:
      PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD
    """
    host = os.environ.get(f"{prefix}HOST", "db")
    port = int(os.environ.get(f"{prefix}PORT", "5432"))
    dbname = os.environ.get(f"{prefix}DATABASE", os.environ.get(f"{prefix}DBNAME", "pqcdb"))
    user = os.environ.get(f"{prefix}USER", "postgres")
    password = os.environ.get(f"{prefix}PASSWORD", "postgres")
    return DbConfig(host=host, port=port, dbname=dbname, user=user, password=password)


def connect(cfg: Optional[DbConfig] = None) -> psycopg.Connection:
    cfg = cfg or config_from_env()
    return psycopg.connect(
        host=cfg.host,
        port=cfg.port,
        dbname=cfg.dbname,
        user=cfg.user,
        password=cfg.password,
        autocommit=False,
    )


# -----------------------------
# Schema / Rows
# -----------------------------

@dataclass(frozen=True)
class DbEncryptedRow:
    id: int
    version: int
    tenant_id: str
    key_id: str

    scheme: str
    payload_cipher: str
    kem_id: Optional[str]
    kdf_id: str
    wrap_id: str
    pq_kem_id: Optional[str]

    aad: bytes
    nonce: bytes
    tag: bytes
    ciphertext: bytes

    wrapped_dek: bytes
    eph_pubkey: bytes

    salt: bytes
    hkdf_info: bytes

    pq_ct: Optional[bytes]

    sig_alg_id: Optional[str]
    signature:  Optional[bytes]
    sig_key_id: Optional[str]

    created_at: Any
    updated_at: Any
    deleted_at: Any


SCHEMA_SQL = """
-- Base table (created once). We intentionally keep it small and then ALTER ADD columns
-- so upgrades work even when a Docker volume already contains the old schema.
CREATE TABLE IF NOT EXISTS encrypted_records (
  id BIGSERIAL PRIMARY KEY
);

-- Add/upgrade columns (idempotent)
ALTER TABLE encrypted_records ADD COLUMN IF NOT EXISTS version        INT  NOT NULL DEFAULT 1;
ALTER TABLE encrypted_records ADD COLUMN IF NOT EXISTS tenant_id      TEXT NOT NULL DEFAULT 'default';
ALTER TABLE encrypted_records ADD COLUMN IF NOT EXISTS key_id         TEXT NOT NULL DEFAULT 'key-v1';

ALTER TABLE encrypted_records ADD COLUMN IF NOT EXISTS scheme         TEXT NOT NULL DEFAULT 'classical';
ALTER TABLE encrypted_records ADD COLUMN IF NOT EXISTS payload_cipher TEXT NOT NULL DEFAULT 'AES-256-GCM';
ALTER TABLE encrypted_records ADD COLUMN IF NOT EXISTS kem_id         TEXT NULL;
ALTER TABLE encrypted_records ADD COLUMN IF NOT EXISTS kdf_id         TEXT NOT NULL DEFAULT 'HKDF-SHA256';
ALTER TABLE encrypted_records ADD COLUMN IF NOT EXISTS wrap_id        TEXT NOT NULL DEFAULT 'AES-KW-RFC3394';
ALTER TABLE encrypted_records ADD COLUMN IF NOT EXISTS pq_kem_id      TEXT NULL;

ALTER TABLE encrypted_records ADD COLUMN IF NOT EXISTS aad            BYTEA NOT NULL DEFAULT ''::bytea;
ALTER TABLE encrypted_records ADD COLUMN IF NOT EXISTS nonce          BYTEA NOT NULL DEFAULT ''::bytea;
ALTER TABLE encrypted_records ADD COLUMN IF NOT EXISTS tag            BYTEA NOT NULL DEFAULT ''::bytea;
ALTER TABLE encrypted_records ADD COLUMN IF NOT EXISTS ciphertext     BYTEA NOT NULL DEFAULT ''::bytea;
ALTER TABLE encrypted_records ADD COLUMN IF NOT EXISTS wrapped_dek    BYTEA NOT NULL DEFAULT ''::bytea;
ALTER TABLE encrypted_records ADD COLUMN IF NOT EXISTS eph_pubkey     BYTEA NOT NULL DEFAULT ''::bytea;
ALTER TABLE encrypted_records ADD COLUMN IF NOT EXISTS salt           BYTEA NOT NULL DEFAULT ''::bytea;
ALTER TABLE encrypted_records ADD COLUMN IF NOT EXISTS hkdf_info      BYTEA NOT NULL DEFAULT ''::bytea;
ALTER TABLE encrypted_records ADD COLUMN IF NOT EXISTS pq_ct          BYTEA NULL;

ALTER TABLE encrypted_records ADD COLUMN IF NOT EXISTS sig_alg_id     TEXT  NULL;
ALTER TABLE encrypted_records ADD COLUMN IF NOT EXISTS signature       BYTEA NULL;
ALTER TABLE encrypted_records ADD COLUMN IF NOT EXISTS sig_key_id      TEXT  NULL;

ALTER TABLE encrypted_records ADD COLUMN IF NOT EXISTS created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW();
ALTER TABLE encrypted_records ADD COLUMN IF NOT EXISTS updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW();
ALTER TABLE encrypted_records ADD COLUMN IF NOT EXISTS deleted_at     TIMESTAMPTZ NULL;

-- Constraints (wrapped in DO blocks so concurrent init doesn't crash startup)
DO $$ BEGIN
  ALTER TABLE encrypted_records
    DROP CONSTRAINT IF EXISTS chk_scheme_values,
    ADD  CONSTRAINT chk_scheme_values CHECK (scheme IN ('classical','hybrid'));
EXCEPTION WHEN others THEN
END $$;

DO $$ BEGIN
  ALTER TABLE encrypted_records
    DROP CONSTRAINT IF EXISTS chk_hybrid_fields,
    ADD  CONSTRAINT chk_hybrid_fields CHECK (
      (scheme = 'classical' AND pq_kem_id IS NULL AND pq_ct IS NULL)
      OR
      (scheme = 'hybrid' AND pq_kem_id IS NOT NULL AND pq_ct IS NOT NULL)
    );
EXCEPTION WHEN others THEN
END $$;

DO $$ BEGIN
  ALTER TABLE encrypted_records
    DROP CONSTRAINT IF EXISTS chk_sig_fields,
    ADD  CONSTRAINT chk_sig_fields CHECK (
      (sig_alg_id IS NULL AND signature IS NULL AND sig_key_id IS NULL)
      OR
      (sig_alg_id IS NOT NULL AND signature IS NOT NULL AND sig_key_id IS NOT NULL)
    );
EXCEPTION WHEN others THEN
END $$;

CREATE INDEX IF NOT EXISTS idx_encrypted_records_tenant_created_at ON encrypted_records(tenant_id, created_at);
CREATE INDEX IF NOT EXISTS idx_encrypted_records_tenant_key_id     ON encrypted_records(tenant_id, key_id);
CREATE INDEX IF NOT EXISTS idx_encrypted_records_scheme            ON encrypted_records(scheme);
CREATE INDEX IF NOT EXISTS idx_encrypted_records_created_at        ON encrypted_records(created_at);

-- Lightweight audit table for observability.
CREATE TABLE IF NOT EXISTS audit_events (
  id         BIGSERIAL PRIMARY KEY,
  ts         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  tenant_id  TEXT NOT NULL,
  record_id  BIGINT NULL,
  event_type TEXT NOT NULL,
  key_id     TEXT NULL,
  scheme     TEXT NULL,
  status     TEXT NOT NULL,
  latency_ms DOUBLE PRECISION NULL,
  detail     TEXT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_events_tenant_ts ON audit_events(tenant_id, ts);
"""


def init_schema(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
    conn.commit()


def audit_event(
    conn: psycopg.Connection,
    *,
    tenant_id: str,
    event_type: str,
    status: str,
    record_id: Optional[int] = None,
    key_id: Optional[str] = None,
    scheme: Optional[str] = None,
    latency_ms: Optional[float] = None,
    detail: Optional[str] = None,
) -> None:
    """Best-effort audit logging; failures should not break the data path."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO audit_events
                  (tenant_id, record_id, event_type, key_id, scheme, status, latency_ms, detail)
                VALUES
                  (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (tenant_id, record_id, event_type, key_id, scheme, status, latency_ms, detail),
            )
        conn.commit()
    except Exception:
        conn.rollback()


def insert_record(
    conn: psycopg.Connection,
    *,
    version: int,
    tenant_id: str,
    key_id: str,
    scheme: str,
    payload_cipher: str,
    kem_id: Optional[str],
    kdf_id: str,
    wrap_id: str,
    pq_kem_id: Optional[str],
    aad: bytes,
    nonce: bytes,
    tag: bytes,
    ciphertext: bytes,
    wrapped_dek: bytes,
    eph_pubkey: bytes,
    salt: bytes,
    hkdf_info: bytes,
    pq_ct: Optional[bytes],
    sig_alg_id: Optional[str] = None,
    signature: Optional[bytes] = None,
    sig_key_id: Optional[str] = None,
) -> Tuple[int, float]:
    """Returns (new_id, db_ms)."""
    t0 = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO encrypted_records
              (version, tenant_id, key_id,
               scheme, payload_cipher, kem_id, kdf_id, wrap_id, pq_kem_id,
               aad, nonce, tag, ciphertext, wrapped_dek, eph_pubkey,
               salt, hkdf_info, pq_ct,
               sig_alg_id, signature, sig_key_id,
               created_at, updated_at, deleted_at)
            VALUES
              (%s, %s, %s,
               %s, %s, %s, %s, %s, %s,
               %s, %s, %s, %s, %s, %s,
               %s, %s, %s,
               %s, %s, %s,
               NOW(), NOW(), NULL)
            RETURNING id
            """,
            (
                version,
                tenant_id,
                key_id,
                scheme,
                payload_cipher,
                kem_id,
                kdf_id,
                wrap_id,
                pq_kem_id,
                aad,
                nonce,
                tag,
                ciphertext,
                wrapped_dek,
                eph_pubkey,
                salt,
                hkdf_info,
                pq_ct,
                sig_alg_id,
                signature,
                sig_key_id,
            ),
        )
        new_id = cur.fetchone()[0]
    conn.commit()
    db_ms = (time.perf_counter() - t0) * 1000.0
    return int(new_id), db_ms


def fetch_record(
    conn: psycopg.Connection,
    *,
    tenant_id: str,
    record_id: int,
    include_deleted: bool = False,
) -> Tuple[DbEncryptedRow, float]:
    """Returns (row, db_ms). Enforces tenant isolation by design."""
    t0 = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
              id,
              version,
              tenant_id,
              key_id,
              scheme,
              payload_cipher,
              kem_id,
              kdf_id,
              wrap_id,
              pq_kem_id,
              aad,
              nonce,
              tag,
              ciphertext,
              wrapped_dek,
              eph_pubkey,
              salt,
              hkdf_info,
              pq_ct,
              sig_alg_id,
              signature,
              sig_key_id,
              created_at,
              updated_at,
              deleted_at
            FROM encrypted_records
            WHERE tenant_id = %s
              AND id = %s
              AND (%s OR deleted_at IS NULL)
            """,
            (tenant_id, record_id, include_deleted),
        )
        r = cur.fetchone()
        if r is None:
            raise KeyError(f"record_id {record_id} not found for tenant '{tenant_id}'")

    db_ms = (time.perf_counter() - t0) * 1000.0

    # psycopg returns BYTEA as memoryview; normalize.
    row = DbEncryptedRow(
        id=int(r[0]),
        version=int(r[1]),
        tenant_id=str(r[2]),
        key_id=str(r[3]),
        scheme=str(r[4]),
        payload_cipher=str(r[5]),
        kem_id=(None if r[6] is None else str(r[6])),
        kdf_id=str(r[7]),
        wrap_id=str(r[8]),
        pq_kem_id=(None if r[9] is None else str(r[9])),
        aad=bytes(r[10]),
        nonce=bytes(r[11]),
        tag=bytes(r[12]),
        ciphertext=bytes(r[13]),
        wrapped_dek=bytes(r[14]),
        eph_pubkey=bytes(r[15]),
        salt=bytes(r[16]),
        hkdf_info=bytes(r[17]),
        pq_ct=(None if r[18] is None else bytes(r[18])),
        sig_alg_id=(None if r[19] is None else str(r[19])),
        signature=(None if r[20] is None else bytes(r[20])),
        sig_key_id=(None if r[21] is None else str(r[21])),
        created_at=r[22],
        updated_at=r[23],
        deleted_at=r[24],
    )
    return row, db_ms


def update_wrap_fields(
    conn: psycopg.Connection,
    *,
    tenant_id: str,
    record_id: int,
    new_key_id: str,
    wrapped_dek: bytes,
    eph_pubkey: bytes,
    salt: bytes,
    pq_ct: Optional[bytes],
    pq_kem_id: Optional[str],
    sig_alg_id: Optional[str] = None,
    signature: Optional[bytes] = None,
    sig_key_id: Optional[str] = None,
) -> float:
    """Update only fields that should change for an envelope 'rewrap' rotation.

    Rotation changes wrapped_dek, eph_pubkey, salt, and pq_ct — all of which
    are covered by the ML-DSA envelope commitment.  Callers must therefore
    either supply a freshly computed signature (sig_alg_id/signature/sig_key_id
    all set) or explicitly clear it (all three left as None, the default).

    Leaving sig_alg_id=None clears the signature columns so the rotated record
    becomes unsigned.  This is the correct safe default when the caller cannot
    re-sign at rotation time.
    """
    t0 = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE encrypted_records
            SET key_id=%s,
                wrapped_dek=%s,
                eph_pubkey=%s,
                salt=%s,
                pq_ct=%s,
                pq_kem_id=%s,
                sig_alg_id=%s,
                signature=%s,
                sig_key_id=%s,
                updated_at=NOW()
            WHERE tenant_id=%s AND id=%s AND deleted_at IS NULL
            """,
            (
                new_key_id, wrapped_dek, eph_pubkey, salt, pq_ct, pq_kem_id,
                sig_alg_id, signature, sig_key_id,
                tenant_id, record_id,
            ),
        )
        if cur.rowcount != 1:
            raise KeyError(f"record_id {record_id} not found (or deleted) for tenant '{tenant_id}'")
    conn.commit()
    return (time.perf_counter() - t0) * 1000.0


def soft_delete_record(conn: psycopg.Connection, *, tenant_id: str, record_id: int) -> float:
    t0 = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE encrypted_records
            SET deleted_at=NOW(), updated_at=NOW()
            WHERE tenant_id=%s AND id=%s AND deleted_at IS NULL
            """,
            (tenant_id, record_id),
        )
        if cur.rowcount != 1:
            raise KeyError(f"record_id {record_id} not found (or already deleted) for tenant '{tenant_id}'")
    conn.commit()
    return (time.perf_counter() - t0) * 1000.0
