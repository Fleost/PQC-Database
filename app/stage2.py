"""
app/stage2.py
=============
Crypto layer: classical and hybrid post-quantum envelope encryption.

Implements two schemes on top of AES-256-GCM:
  classical — X25519-ECDH + HKDF-SHA256 + AES-KW
  hybrid    — X25519-ECDH + ML-KEM-768 + HKDF-SHA256 + AES-KW

Public API:
  encrypt(plaintext, scheme=..., ...)  -> EncryptedRecord
  decrypt(record, ...)                 -> (plaintext, metrics)
  generate_classical_recipient_keys()  -> ClassicalRecipientKeys
  generate_hybrid_recipient_keys(...)  -> HybridRecipientKeys
  envelope_sizes(record)               -> Dict[str, int]

Run the built-in smoke test:
  python -m app.stage2
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Protocol, Tuple
import os
import time

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.asymmetric import x25519
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes, keywrap
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat


# -----------------------------
# IDs / constants (crypto agility)
# -----------------------------

RECORD_VERSION_V1 = 1

PAYLOAD_CIPHER_ID_AES_256_GCM = "AES-256-GCM"
KEM_ID_X25519 = "X25519-ECDH"
KDF_ID_HKDF_SHA256 = "HKDF-SHA256"
WRAP_ID_AES_KW = "AES-KW-RFC3394"

DEFAULT_HKDF_INFO_CLASSICAL = b"dek-wrap:v1:classical"
DEFAULT_HKDF_INFO_HYBRID = b"dek-wrap:v1:hybrid"

# oqs typically supports "ML-KEM-768" (newer) or "Kyber768" (older).
DEFAULT_PQ_KEM_ID = "ML-KEM-768"


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


# -----------------------------
# PQ KEM backend (oqs)
# -----------------------------

class PQKEM(Protocol):
    kem_id: str

    def keygen(self) -> Tuple[bytes, bytes]:
        ...

    def encaps(self, public_key: bytes) -> Tuple[bytes, bytes]:
        ...

    def decaps(self, secret_key: bytes, ciphertext: bytes) -> bytes:
        ...


class OQSKEMBackend:
    """
    oqs-python backend.
    In Docker, we install liboqs + pip install oqs, so this should work.
    """
    def __init__(self, kem_id: str = DEFAULT_PQ_KEM_ID):
        import oqs  # provided by `pip install oqs`

        self._oqs = oqs
        self.kem_id = kem_id

        # Try both ML-KEM-* and legacy Kyber* names automatically
        self._name_candidates = [kem_id]
        if kem_id.upper().startswith("ML-KEM-"):
            bits = kem_id.split("-")[-1]
            self._name_candidates.append(f"Kyber{bits}")

    def _new_kem(self):
        last_err = None
        for name in self._name_candidates:
            try:
                return self._oqs.KeyEncapsulation(name)
            except Exception as e:
                last_err = e
        raise ValueError(
            f"oqs: could not create KEM for {self.kem_id} "
            f"(tried {self._name_candidates})."
        ) from last_err

    def keygen(self) -> Tuple[bytes, bytes]:
        with self._new_kem() as kem:
            pk = kem.generate_keypair()
            sk = kem.export_secret_key()
            return pk, sk

    def encaps(self, public_key: bytes) -> Tuple[bytes, bytes]:
        with self._new_kem() as kem:
            ct, ss = kem.encap_secret(public_key)
            return ct, ss

    def decaps(self, secret_key: bytes, ciphertext: bytes) -> bytes:
        # oqs-python expects the secret key at construction time (no import_secret_key)
        last_err = None
        for name in self._name_candidates:
            try:
                with self._oqs.KeyEncapsulation(name, secret_key) as kem:
                    return kem.decap_secret(ciphertext)
            except TypeError:
                # some builds may require keyword arg
                try:
                    with self._oqs.KeyEncapsulation(name, secret_key=secret_key) as kem:
                        return kem.decap_secret(ciphertext)
                except Exception as e:
                    last_err = e
            except Exception as e:
                last_err = e
        raise ValueError(f"oqs: could not decaps for {self.kem_id} (tried {self._name_candidates}).") from last_err


# -----------------------------
# Key bundles
# -----------------------------

@dataclass(frozen=True)
class ClassicalRecipientKeys:
    sk: x25519.X25519PrivateKey
    pk: x25519.X25519PublicKey


def generate_classical_recipient_keys() -> ClassicalRecipientKeys:
    sk = x25519.X25519PrivateKey.generate()
    pk = sk.public_key()
    return ClassicalRecipientKeys(sk=sk, pk=pk)


@dataclass(frozen=True)
class HybridRecipientKeys:
    classical: ClassicalRecipientKeys
    pq_public_key: bytes
    pq_secret_key: bytes
    pq_kem_id: str

    @property
    def pq_sk(self) -> bytes:
        return self.pq_secret_key


def generate_hybrid_recipient_keys(pq_kem_id: str = DEFAULT_PQ_KEM_ID) -> HybridRecipientKeys:
    pq = OQSKEMBackend(pq_kem_id)
    pq_pk, pq_sk = pq.keygen()
    return HybridRecipientKeys(
        classical=generate_classical_recipient_keys(),
        pq_public_key=pq_pk,
        pq_secret_key=pq_sk,
        pq_kem_id=pq.kem_id,
    )


# -----------------------------
# Record format
# -----------------------------

@dataclass
class EncryptedRecord:
    version: int
    scheme: str  # "classical" | "hybrid"

    payload_cipher: str
    kem_id: str
    kdf_id: str
    wrap_id: str

    ciphertext: bytes
    nonce: bytes
    tag: bytes
    aad: bytes

    wrapped_dek: bytes
    ephemeral_pubkey: bytes  # raw 32 bytes (X25519 public key)
    salt: bytes              # per-record HKDF salt
    hkdf_info: bytes         # per-record HKDF info (stored to prevent mismatch)

    # Hybrid-only fields:
    pq_kem_id: Optional[str] = None
    pq_ciphertext: Optional[bytes] = None

    # Optional timing metrics:
    metrics: Optional[Dict[str, float]] = None


# -----------------------------
# Helpers
# -----------------------------

def _serialize_pubkey(pk: x25519.X25519PublicKey) -> bytes:
    raw = pk.public_bytes(Encoding.Raw, PublicFormat.Raw)
    if len(raw) != 32:
        raise ValueError(f"Unexpected X25519 public key length: {len(raw)}")
    return raw


def _deserialize_pubkey(raw: bytes) -> x25519.X25519PublicKey:
    if len(raw) != 32:
        raise ValueError(f"Invalid X25519 public key length: {len(raw)}")
    return x25519.X25519PublicKey.from_public_bytes(raw)


def _hkdf_derive_kek(ikm: bytes, *, salt: bytes, info: bytes, length: int = 32) -> bytes:
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=length,
        salt=salt,
        info=info,
    )
    return hkdf.derive(ikm)


def envelope_sizes(rec: EncryptedRecord) -> Dict[str, int]:
    """Return the byte length of every stored field in an EncryptedRecord.

    Useful for storage-overhead analysis: compare these sizes across schemes
    to quantify the extra bytes introduced by the PQ KEM layer.
    """
    pq_ct_len = len(rec.pq_ciphertext) if rec.pq_ciphertext else 0
    # "header" = everything stored except the payload ciphertext and PQ ciphertext
    header_bytes = (
        len(rec.nonce)
        + len(rec.tag)
        + len(rec.wrapped_dek)
        + len(rec.ephemeral_pubkey)
        + len(rec.salt)
        + len(rec.hkdf_info)
        + len(rec.aad)
    )
    total = len(rec.ciphertext) + header_bytes + pq_ct_len
    return {
        "ciphertext_bytes":    len(rec.ciphertext),
        "nonce_bytes":         len(rec.nonce),
        "tag_bytes":           len(rec.tag),
        "wrapped_dek_bytes":   len(rec.wrapped_dek),
        "eph_pubkey_bytes":    len(rec.ephemeral_pubkey),
        "salt_bytes":          len(rec.salt),
        "hkdf_info_bytes":     len(rec.hkdf_info),
        "aad_bytes":           len(rec.aad),
        "pq_ct_bytes":         pq_ct_len,
        "header_bytes":        header_bytes,
        "overhead_bytes":      header_bytes + pq_ct_len,
        "total_stored_bytes":  total,
    }


def _validate_record(record: EncryptedRecord) -> None:
    if record.version != RECORD_VERSION_V1:
        raise ValueError(f"Unsupported record version: {record.version}")
    if record.scheme not in ("classical", "hybrid"):
        raise ValueError(f"Unsupported scheme: {record.scheme}")
    if record.payload_cipher != PAYLOAD_CIPHER_ID_AES_256_GCM:
        raise ValueError("Unsupported payload cipher")
    if record.kem_id != KEM_ID_X25519:
        raise ValueError("Unsupported kem_id")
    if record.kdf_id != KDF_ID_HKDF_SHA256:
        raise ValueError("Unsupported kdf_id")
    if record.wrap_id != WRAP_ID_AES_KW:
        raise ValueError("Unsupported wrap_id")
    if len(record.nonce) != 12:
        raise ValueError("Invalid nonce length")
    if len(record.tag) != 16:
        raise ValueError("Invalid tag length")
    if len(record.ephemeral_pubkey) != 32:
        raise ValueError("Invalid ephemeral pubkey length")
    if len(record.salt) < 16:
        raise ValueError("Salt too short (>=16 recommended)")
    if not record.hkdf_info:
        raise ValueError("hkdf_info required")

    if record.scheme == "classical":
        if record.pq_kem_id is not None or record.pq_ciphertext is not None:
            raise ValueError("Classical record must not include PQ fields")
    else:
        if not record.pq_kem_id or record.pq_ciphertext is None:
            raise ValueError("Hybrid record missing PQ fields")


# -----------------------------
# Classical (Stage 1 style)
# -----------------------------

def classical_encrypt_v1(
    plaintext: bytes,
    *,
    recipient_pk: x25519.X25519PublicKey,
    aad: bytes = b"",
    salt_len: int = 32,
    hkdf_info: bytes = DEFAULT_HKDF_INFO_CLASSICAL,
    collect_metrics: bool = True,
) -> EncryptedRecord:
    metrics: Dict[str, float] = {}

    t0 = _now_ms()
    dek = os.urandom(32)
    if collect_metrics:
        metrics["t_dek_gen_ms"] = _now_ms() - t0

    t1 = _now_ms()
    nonce = os.urandom(12)
    aesgcm = AESGCM(dek)
    ct_and_tag = aesgcm.encrypt(nonce, plaintext, aad)
    ciphertext, tag = ct_and_tag[:-16], ct_and_tag[-16:]
    if collect_metrics:
        metrics["t_aes_gcm_ms"] = _now_ms() - t1

    t2 = _now_ms()
    eph_sk = x25519.X25519PrivateKey.generate()
    eph_pk = eph_sk.public_key()
    eph_raw = _serialize_pubkey(eph_pk)
    if collect_metrics:
        metrics["t_eph_keygen_ms"] = _now_ms() - t2

    t3 = _now_ms()
    ss_ecc = eph_sk.exchange(recipient_pk)
    if collect_metrics:
        metrics["t_ecdh_ms"] = _now_ms() - t3

    t4 = _now_ms()
    salt = os.urandom(salt_len)
    kek = _hkdf_derive_kek(ss_ecc, salt=salt, info=hkdf_info, length=32)
    if collect_metrics:
        metrics["t_hkdf_ms"] = _now_ms() - t4

    t5 = _now_ms()
    wrapped_dek = keywrap.aes_key_wrap(kek, dek)
    if collect_metrics:
        metrics["t_wrap_ms"] = _now_ms() - t5
        metrics["t_total_crypto_ms"] = sum(
            metrics[k] for k in ("t_dek_gen_ms", "t_aes_gcm_ms", "t_eph_keygen_ms", "t_ecdh_ms", "t_hkdf_ms", "t_wrap_ms")
        )

    rec = EncryptedRecord(
        version=RECORD_VERSION_V1,
        scheme="classical",
        payload_cipher=PAYLOAD_CIPHER_ID_AES_256_GCM,
        kem_id=KEM_ID_X25519,
        kdf_id=KDF_ID_HKDF_SHA256,
        wrap_id=WRAP_ID_AES_KW,
        ciphertext=ciphertext,
        nonce=nonce,
        tag=tag,
        aad=aad,
        wrapped_dek=wrapped_dek,
        ephemeral_pubkey=eph_raw,
        salt=salt,
        hkdf_info=hkdf_info,
        pq_kem_id=None,
        pq_ciphertext=None,
        metrics=metrics if collect_metrics else None,
    )
    _validate_record(rec)
    return rec


def classical_decrypt_v1(
    rec: EncryptedRecord,
    *,
    recipient_sk: x25519.X25519PrivateKey,
    collect_metrics: bool = True,
) -> Tuple[bytes, Dict[str, float]]:
    _validate_record(rec)
    if rec.scheme != "classical":
        raise ValueError("Record is not classical")

    metrics: Dict[str, float] = {}

    t0 = _now_ms()
    eph_pk = _deserialize_pubkey(rec.ephemeral_pubkey)
    ss_ecc = recipient_sk.exchange(eph_pk)
    if collect_metrics:
        metrics["t_ecdh_ms"] = _now_ms() - t0

    t1 = _now_ms()
    kek = _hkdf_derive_kek(ss_ecc, salt=rec.salt, info=rec.hkdf_info, length=32)
    if collect_metrics:
        metrics["t_hkdf_ms"] = _now_ms() - t1

    t2 = _now_ms()
    dek = keywrap.aes_key_unwrap(kek, rec.wrapped_dek)
    if collect_metrics:
        metrics["t_unwrap_ms"] = _now_ms() - t2

    t3 = _now_ms()
    aesgcm = AESGCM(dek)
    plaintext = aesgcm.decrypt(rec.nonce, rec.ciphertext + rec.tag, rec.aad)
    if collect_metrics:
        metrics["t_aes_gcm_ms"] = _now_ms() - t3
        metrics["t_total_crypto_ms"] = sum(
            metrics[k] for k in ("t_ecdh_ms", "t_hkdf_ms", "t_unwrap_ms", "t_aes_gcm_ms")
        )

    return plaintext, metrics


# -----------------------------
# Hybrid (Stage 2)
# -----------------------------

def hybrid_encrypt_v1(
    plaintext: bytes,
    *,
    recipient_classical_pk: x25519.X25519PublicKey,
    recipient_pq_pk: bytes,
    pq_kem_id: str = DEFAULT_PQ_KEM_ID,
    aad: bytes = b"",
    salt_len: int = 32,
    hkdf_info: bytes = DEFAULT_HKDF_INFO_HYBRID,
    collect_metrics: bool = True,
) -> EncryptedRecord:
    pq = OQSKEMBackend(pq_kem_id)
    metrics: Dict[str, float] = {}

    t0 = _now_ms()
    dek = os.urandom(32)
    if collect_metrics:
        metrics["t_dek_gen_ms"] = _now_ms() - t0

    t1 = _now_ms()
    nonce = os.urandom(12)
    aesgcm = AESGCM(dek)
    ct_and_tag = aesgcm.encrypt(nonce, plaintext, aad)
    ciphertext, tag = ct_and_tag[:-16], ct_and_tag[-16:]
    if collect_metrics:
        metrics["t_aes_gcm_ms"] = _now_ms() - t1

    t2 = _now_ms()
    eph_sk = x25519.X25519PrivateKey.generate()
    eph_pk = eph_sk.public_key()
    eph_raw = _serialize_pubkey(eph_pk)
    if collect_metrics:
        metrics["t_eph_keygen_ms"] = _now_ms() - t2

    t3 = _now_ms()
    ss_ecc = eph_sk.exchange(recipient_classical_pk)
    if collect_metrics:
        metrics["t_ecdh_ms"] = _now_ms() - t3

    t4 = _now_ms()
    pq_ct, ss_pq = pq.encaps(recipient_pq_pk)
    if collect_metrics:
        metrics["t_pq_encaps_ms"] = _now_ms() - t4
        metrics["pq_ct_bytes"] = float(len(pq_ct))
        metrics["pq_ss_bytes"] = float(len(ss_pq))

    t5 = _now_ms()
    salt = os.urandom(salt_len)
    ikm = ss_ecc + ss_pq
    kek = _hkdf_derive_kek(ikm, salt=salt, info=hkdf_info, length=32)
    if collect_metrics:
        metrics["t_hkdf_ms"] = _now_ms() - t5

    t6 = _now_ms()
    wrapped_dek = keywrap.aes_key_wrap(kek, dek)
    if collect_metrics:
        metrics["t_wrap_ms"] = _now_ms() - t6
        metrics["t_total_crypto_ms"] = sum(
            metrics[k] for k in (
                "t_dek_gen_ms", "t_aes_gcm_ms", "t_eph_keygen_ms",
                "t_ecdh_ms", "t_pq_encaps_ms", "t_hkdf_ms", "t_wrap_ms"
            )
        )

    rec = EncryptedRecord(
        version=RECORD_VERSION_V1,
        scheme="hybrid",
        payload_cipher=PAYLOAD_CIPHER_ID_AES_256_GCM,
        kem_id=KEM_ID_X25519,
        kdf_id=KDF_ID_HKDF_SHA256,
        wrap_id=WRAP_ID_AES_KW,
        ciphertext=ciphertext,
        nonce=nonce,
        tag=tag,
        aad=aad,
        wrapped_dek=wrapped_dek,
        ephemeral_pubkey=eph_raw,
        salt=salt,
        hkdf_info=hkdf_info,
        pq_kem_id=pq.kem_id,
        pq_ciphertext=pq_ct,
        metrics=metrics if collect_metrics else None,
    )
    _validate_record(rec)
    return rec


def hybrid_decrypt_v1(
    rec: EncryptedRecord,
    *,
    recipient_classical_sk: x25519.X25519PrivateKey,
    recipient_pq_sk: bytes,
    collect_metrics: bool = True,
) -> Tuple[bytes, Dict[str, float]]:
    _validate_record(rec)
    if rec.scheme != "hybrid":
        raise ValueError("Record is not hybrid")
    if rec.pq_kem_id is None or rec.pq_ciphertext is None:
        raise ValueError("Hybrid record missing PQ fields")

    pq = OQSKEMBackend(rec.pq_kem_id)
    metrics: Dict[str, float] = {}

    t0 = _now_ms()
    eph_pk = _deserialize_pubkey(rec.ephemeral_pubkey)
    ss_ecc = recipient_classical_sk.exchange(eph_pk)
    if collect_metrics:
        metrics["t_ecdh_ms"] = _now_ms() - t0

    t1 = _now_ms()
    ss_pq = pq.decaps(recipient_pq_sk, rec.pq_ciphertext)
    if collect_metrics:
        metrics["t_pq_decaps_ms"] = _now_ms() - t1

    t2 = _now_ms()
    ikm = ss_ecc + ss_pq
    kek = _hkdf_derive_kek(ikm, salt=rec.salt, info=rec.hkdf_info, length=32)
    if collect_metrics:
        metrics["t_hkdf_ms"] = _now_ms() - t2

    t3 = _now_ms()
    dek = keywrap.aes_key_unwrap(kek, rec.wrapped_dek)
    if collect_metrics:
        metrics["t_unwrap_ms"] = _now_ms() - t3

    t4 = _now_ms()
    aesgcm = AESGCM(dek)
    plaintext = aesgcm.decrypt(rec.nonce, rec.ciphertext + rec.tag, rec.aad)
    if collect_metrics:
        metrics["t_aes_gcm_ms"] = _now_ms() - t4
        metrics["t_total_crypto_ms"] = sum(
            metrics[k] for k in ("t_ecdh_ms", "t_pq_decaps_ms", "t_hkdf_ms", "t_unwrap_ms", "t_aes_gcm_ms")
        )

    return plaintext, metrics


# -----------------------------
# Unified API
# -----------------------------

def encrypt(
    plaintext: bytes,
    *,
    scheme: str,
    recipient_classical_pk: x25519.X25519PublicKey,
    recipient_pq_pk: Optional[bytes] = None,
    pq_kem_id: str = DEFAULT_PQ_KEM_ID,
    aad: bytes = b"",
    collect_metrics: bool = True,
) -> EncryptedRecord:
    if scheme == "classical":
        return classical_encrypt_v1(
            plaintext,
            recipient_pk=recipient_classical_pk,
            aad=aad,
            collect_metrics=collect_metrics,
        )
    if scheme == "hybrid":
        if recipient_pq_pk is None:
            raise ValueError("recipient_pq_pk required for hybrid encryption")
        return hybrid_encrypt_v1(
            plaintext,
            recipient_classical_pk=recipient_classical_pk,
            recipient_pq_pk=recipient_pq_pk,
            pq_kem_id=pq_kem_id,
            aad=aad,
            collect_metrics=collect_metrics,
        )
    raise ValueError(f"Unknown scheme: {scheme}")


def decrypt(
    rec: EncryptedRecord,
    *,
    recipient_classical_sk: x25519.X25519PrivateKey,
    recipient_pq_sk: Optional[bytes] = None,
    collect_metrics: bool = True,
) -> Tuple[bytes, Dict[str, float]]:
    if rec.scheme == "classical":
        return classical_decrypt_v1(rec, recipient_sk=recipient_classical_sk, collect_metrics=collect_metrics)
    if rec.scheme == "hybrid":
        if recipient_pq_sk is None:
            raise ValueError("recipient_pq_sk required for hybrid decryption")
        return hybrid_decrypt_v1(
            rec,
            recipient_classical_sk=recipient_classical_sk,
            recipient_pq_sk=recipient_pq_sk,
            collect_metrics=collect_metrics,
        )
    raise ValueError(f"Unsupported scheme: {rec.scheme}")


# -----------------------------
# Smoke test
# -----------------------------

def _smoke_test_stage2() -> None:
    print("== Stage 2 smoke test ==")

    plaintext = b"genomic payload bytes (toy example)"
    aad = b"patient_id=123|record_type=variants|v=1"

    # classical test
    classical_recipient = generate_classical_recipient_keys()
    rec_c = encrypt(
        plaintext,
        scheme="classical",
        recipient_classical_pk=classical_recipient.pk,
        aad=aad,
        collect_metrics=True,
    )
    out_c, _ = decrypt(rec_c, recipient_classical_sk=classical_recipient.sk)
    assert out_c == plaintext
    print("OK: classical encrypt/decrypt")

    # hybrid test
    hybrid_recipient = generate_hybrid_recipient_keys(DEFAULT_PQ_KEM_ID)
    rec_h = encrypt(
        plaintext,
        scheme="hybrid",
        recipient_classical_pk=hybrid_recipient.classical.pk,
        recipient_pq_pk=hybrid_recipient.pq_public_key,
        pq_kem_id=hybrid_recipient.pq_kem_id,
        aad=aad,
        collect_metrics=True,
    )
    out_h, _ = decrypt(
        rec_h,
        recipient_classical_sk=hybrid_recipient.classical.sk,
        recipient_pq_sk=hybrid_recipient.pq_secret_key,
    )
    assert out_h == plaintext
    print("OK: hybrid encrypt/decrypt")

    # tamper test
    tampered = EncryptedRecord(**{**rec_h.__dict__, "aad": rec_h.aad[:-1] + b"X"})
    try:
        decrypt(
            tampered,
            recipient_classical_sk=hybrid_recipient.classical.sk,
            recipient_pq_sk=hybrid_recipient.pq_secret_key,
        )
        raise AssertionError("Tampering was NOT detected (unexpected).")
    except InvalidTag:
        print("OK: tampering detected (InvalidTag).")

    print(f"Ciphertext bytes: {len(rec_h.ciphertext)}")
    print(f"Wrapped DEK bytes: {len(rec_h.wrapped_dek)}")
    print(f"Ephemeral pubkey bytes: {len(rec_h.ephemeral_pubkey)}")
    print(f"Salt bytes: {len(rec_h.salt)}")
    print(f"PQ ct bytes: {len(rec_h.pq_ciphertext or b'')}")
    if rec_h.metrics:
        print("Hybrid metrics (ms):")
        for k, v in rec_h.metrics.items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    _smoke_test_stage2()
