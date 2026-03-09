"""
Service-layer API for the PQC DB eval project.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from cryptography.hazmat.primitives import keywrap
from cryptography.hazmat.primitives.asymmetric import x25519

from . import db
from . import stage2


# -----------------------------
# Context / policy
# -----------------------------

@dataclass(frozen=True)
class ServiceContext:
    tenant_id: str
    key_id: str
    version: int = 1


# Backwards-compatible alias expected by stage3.py
KeyContext = ServiceContext


# -----------------------------
# Internal helpers
# -----------------------------

def _row_to_record(row: db.DbEncryptedRow) -> stage2.EncryptedRecord:
    """Convert a DB row into a stage2 EncryptedRecord for decryption."""
    return stage2.EncryptedRecord(
        version=row.version,
        scheme=row.scheme,
        payload_cipher=row.payload_cipher,
        kem_id=row.kem_id,
        kdf_id=row.kdf_id,
        wrap_id=row.wrap_id,
        ciphertext=row.ciphertext,
        nonce=row.nonce,
        tag=row.tag,
        aad=row.aad,
        wrapped_dek=row.wrapped_dek,
        ephemeral_pubkey=row.eph_pubkey,
        salt=row.salt,
        hkdf_info=row.hkdf_info,
        pq_kem_id=row.pq_kem_id,
        pq_ciphertext=row.pq_ct,
    )


# -----------------------------
# Public service API
# -----------------------------

def put_record(
    conn,
    *,
    ctx: ServiceContext,
    plaintext: bytes,
    aad: bytes,
    scheme: str,
    recipient_keys: Any,
    pq_kem_id: Optional[str] = None,
) -> Tuple[int, Dict[str, float]]:
    """Encrypt plaintext and insert into the DB. Returns (record_id, timing_dict)."""
    t0 = time.perf_counter()

    if scheme == "classical":
        rec = stage2.encrypt(
            plaintext,
            scheme="classical",
            recipient_classical_pk=recipient_keys.pk,
            aad=aad,
        )
    elif scheme == "hybrid":
        rec = stage2.encrypt(
            plaintext,
            scheme="hybrid",
            recipient_classical_pk=recipient_keys.classical.pk,
            recipient_pq_pk=recipient_keys.pq_public_key,
            pq_kem_id=pq_kem_id or stage2.DEFAULT_PQ_KEM_ID,
            aad=aad,
        )
    else:
        raise ValueError(f"Unknown scheme: {scheme!r}")

    t_encrypt = (time.perf_counter() - t0) * 1000.0
    env_sizes = stage2.envelope_sizes(rec)

    # Time db_insert to include the audit write — both are committed DB operations
    # in the write path and together account for the full "store" cost.
    t_db = time.perf_counter()
    record_id, _ = db.insert_record(
        conn,
        version=rec.version,
        tenant_id=ctx.tenant_id,
        key_id=ctx.key_id,
        scheme=rec.scheme,
        payload_cipher=rec.payload_cipher,
        kem_id=rec.kem_id,
        kdf_id=rec.kdf_id,
        wrap_id=rec.wrap_id,
        pq_kem_id=rec.pq_kem_id,
        aad=rec.aad,
        nonce=rec.nonce,
        tag=rec.tag,
        ciphertext=rec.ciphertext,
        wrapped_dek=rec.wrapped_dek,
        eph_pubkey=rec.ephemeral_pubkey,
        salt=rec.salt,
        hkdf_info=rec.hkdf_info,
        pq_ct=rec.pq_ciphertext,
    )
    db.audit_event(
        conn,
        tenant_id=ctx.tenant_id,
        record_id=record_id,
        event_type="put",
        key_id=ctx.key_id,
        scheme=scheme,
        status="ok",
    )
    db_ms = (time.perf_counter() - t_db) * 1000.0

    total_ms = (time.perf_counter() - t0) * 1000.0
    timings = {
        "encrypt": t_encrypt,
        "db_insert": db_ms,
        "total": total_ms,
        "envelope": env_sizes,
        "encrypt_ops": rec.metrics or {},
    }
    return record_id, timings


def get_record(
    conn,
    *,
    tenant_id: str,
    record_id: int,
    recipient_keys: Any,
    pq_sk: Optional[bytes] = None,
    include_deleted: bool = False,
) -> Tuple[bytes, Dict[str, float]]:
    """Fetch and decrypt a record. Returns (plaintext, timing_dict)."""
    t0 = time.perf_counter()

    row, db_ms = db.fetch_record(
        conn, tenant_id=tenant_id, record_id=record_id, include_deleted=include_deleted
    )
    rec = _row_to_record(row)

    t2 = time.perf_counter()
    if row.scheme == "classical":
        pt, decrypt_ops = stage2.decrypt(rec, recipient_classical_sk=recipient_keys.sk)
    else:
        pt, decrypt_ops = stage2.decrypt(
            rec,
            recipient_classical_sk=recipient_keys.classical.sk,
            recipient_pq_sk=pq_sk,
        )
    t_decrypt = (time.perf_counter() - t2) * 1000.0

    timings = {"db_fetch": db_ms, "decrypt": t_decrypt, "decrypt_ops": decrypt_ops}
    return pt, timings


def delete_record(conn, *, tenant_id: str, record_id: int) -> None:
    """Soft-delete a record."""
    db.soft_delete_record(conn, tenant_id=tenant_id, record_id=record_id)
    db.audit_event(
        conn,
        tenant_id=tenant_id,
        record_id=record_id,
        event_type="delete",
        key_id=None,
        scheme=None,
        status="ok",
    )


def rotate_record_key(
    conn,
    *,
    tenant_id: str,
    record_id: int,
    new_key_id: str,
    recipient_keys: Any,
    pq_sk: Optional[bytes] = None,
) -> Dict[str, Any]:
    """
    Rewrap the DEK under a new ephemeral key (ciphertext unchanged).
    Returns {"mode": "rewrap"}.
    """
    row, _ = db.fetch_record(conn, tenant_id=tenant_id, record_id=record_id)
    rec = _row_to_record(row)

    # Step 1: Derive old KEK and unwrap DEK
    eph_pk = stage2._deserialize_pubkey(rec.ephemeral_pubkey)
    if row.scheme == "classical":
        ss_ecc = recipient_keys.sk.exchange(eph_pk)
        old_kek = stage2._hkdf_derive_kek(ss_ecc, salt=rec.salt, info=rec.hkdf_info, length=32)
    else:
        ss_ecc = recipient_keys.classical.sk.exchange(eph_pk)
        pq_backend = stage2.OQSKEMBackend(rec.pq_kem_id)
        ss_pq = pq_backend.decaps(pq_sk, rec.pq_ciphertext)
        old_kek = stage2._hkdf_derive_kek(
            ss_ecc + ss_pq, salt=rec.salt, info=rec.hkdf_info, length=32
        )

    dek = keywrap.aes_key_unwrap(old_kek, rec.wrapped_dek)

    # Step 2: Wrap DEK under new ephemeral key (same recipient public keys)
    new_salt = os.urandom(16)
    new_eph_sk = x25519.X25519PrivateKey.generate()
    new_eph_pk_raw = stage2._serialize_pubkey(new_eph_sk.public_key())

    if row.scheme == "classical":
        ss_ecc_new = new_eph_sk.exchange(recipient_keys.pk)
        new_kek = stage2._hkdf_derive_kek(
            ss_ecc_new, salt=new_salt, info=rec.hkdf_info, length=32
        )
        new_wrapped_dek = keywrap.aes_key_wrap(new_kek, dek)
        new_pq_ct = None
        new_pq_kem_id = None
    else:
        ss_ecc_new = new_eph_sk.exchange(recipient_keys.classical.pk)
        pq_backend2 = stage2.OQSKEMBackend(rec.pq_kem_id)
        new_pq_ct, ss_pq_new = pq_backend2.encaps(recipient_keys.pq_public_key)
        new_kek = stage2._hkdf_derive_kek(
            ss_ecc_new + ss_pq_new, salt=new_salt, info=rec.hkdf_info, length=32
        )
        new_wrapped_dek = keywrap.aes_key_wrap(new_kek, dek)
        new_pq_kem_id = rec.pq_kem_id

    db.update_wrap_fields(
        conn,
        tenant_id=tenant_id,
        record_id=record_id,
        new_key_id=new_key_id,
        wrapped_dek=new_wrapped_dek,
        eph_pubkey=new_eph_pk_raw,
        salt=new_salt,
        pq_ct=new_pq_ct,
        pq_kem_id=new_pq_kem_id,
    )
    db.audit_event(
        conn,
        tenant_id=tenant_id,
        record_id=record_id,
        event_type="rotate",
        key_id=new_key_id,
        scheme=row.scheme,
        status="ok",
        detail="rewrap",
    )
    return {"mode": "rewrap"}
