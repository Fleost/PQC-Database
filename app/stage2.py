"""
app/stage2.py
=============
Crypto layer: classical and hybrid post-quantum envelope encryption.

Implements two schemes on top of AES-256-GCM:
  classical — X25519-ECDH + HKDF-SHA256 + AES-KW
  hybrid    — X25519-ECDH + ML-KEM-{512,768,1024} + HKDF-SHA256 + AES-KW

Supported PQ KEM parameter sets (NIST FIPS 203 / ML-KEM):
  ML-KEM-512   — NIST security level 1
  ML-KEM-768   — NIST security level 3 (default)
  ML-KEM-1024  — NIST security level 5

Public API:
  encrypt(plaintext, scheme=..., ...)  -> EncryptedRecord
  decrypt(record, ...)                 -> (plaintext, metrics)
  generate_classical_recipient_keys()  -> ClassicalRecipientKeys
  generate_hybrid_recipient_keys(...)  -> HybridRecipientKeys
  envelope_sizes(record)               -> Dict[str, int]
  validate_pq_kem_id(kem_id)           -> None  (raises ValueError if unsupported)

Run the built-in smoke test:
  python -m app.stage2
"""
from __future__ import annotations

import dataclasses
import struct
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

# Supported ML-KEM parameter sets (NIST FIPS 203).
# oqs typically supports "ML-KEM-NNN" (newer) or "KyberNNN" (older liboqs builds).
SUPPORTED_PQ_KEM_IDS = ["ML-KEM-512", "ML-KEM-768", "ML-KEM-1024"]
DEFAULT_PQ_KEM_ID = "ML-KEM-768"

# Supported ML-DSA parameter sets (NIST FIPS 204).
# Historical names: Dilithium2 / Dilithium3 / Dilithium5 (older liboqs builds).
SUPPORTED_PQ_SIG_IDS = ["ML-DSA-44", "ML-DSA-65", "ML-DSA-87"]
DEFAULT_PQ_SIG_ID: Optional[str] = None  # None = unsigned mode

# Domain separator for the canonical envelope commitment (prevents cross-context misuse).
COMMITMENT_DOMAIN_V1 = b"pqc-db-eval:envelope-commitment:v1"


class SigVerificationError(Exception):
    """Raised when ML-DSA signature verification fails on an envelope."""


def validate_pq_kem_id(kem_id: str) -> None:
    """Raise ValueError if kem_id is not a supported ML-KEM parameter set."""
    if kem_id not in SUPPORTED_PQ_KEM_IDS:
        raise ValueError(
            f"Unsupported PQ KEM identifier: {kem_id!r}. "
            f"Must be one of: {', '.join(SUPPORTED_PQ_KEM_IDS)}"
        )


def validate_pq_sig_id(sig_id: str) -> None:
    """Raise ValueError if sig_id is not a supported ML-DSA parameter set."""
    if sig_id not in SUPPORTED_PQ_SIG_IDS:
        raise ValueError(
            f"Unsupported PQ signature identifier: {sig_id!r}. "
            f"Must be one of: {', '.join(SUPPORTED_PQ_SIG_IDS)}"
        )


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
# PQ Signature backend (oqs)
# -----------------------------

class OQSSigBackend:
    """
    oqs-python backend for ML-DSA digital signatures (NIST FIPS 204).

    Handles standardized "ML-DSA-*" names and falls back to historical
    Dilithium{2,3,5} names for older liboqs builds.
    """
    def __init__(self, sig_id: str = "ML-DSA-65"):
        import oqs
        self._oqs = oqs
        self.sig_id = sig_id

        # Primary: standardized ML-DSA names (liboqs >= 0.12.0).
        # Fallback: historical Dilithium names used by older liboqs builds.
        self._name_candidates = [sig_id]
        if sig_id == "ML-DSA-44":
            self._name_candidates.append("Dilithium2")
        elif sig_id == "ML-DSA-65":
            self._name_candidates.append("Dilithium3")
        elif sig_id == "ML-DSA-87":
            self._name_candidates.append("Dilithium5")

    def _new_sig(self, secret_key: Optional[bytes] = None):
        last_err = None
        for name in self._name_candidates:
            try:
                if secret_key is not None:
                    return self._oqs.Signature(name, secret_key)
                return self._oqs.Signature(name)
            except Exception as e:
                last_err = e
        raise ValueError(
            f"oqs: could not create Signature for {self.sig_id} "
            f"(tried {self._name_candidates})."
        ) from last_err

    def keygen(self) -> Tuple[bytes, bytes]:
        """Generate a key pair. Returns (verify_key, signing_key)."""
        with self._new_sig() as sig:
            vk = sig.generate_keypair()
            sk = sig.export_secret_key()
            return vk, sk

    def sign(self, message: bytes, signing_key: bytes) -> bytes:
        """Sign message with the given secret key."""
        last_err = None
        for name in self._name_candidates:
            try:
                with self._oqs.Signature(name, signing_key) as sig:
                    return sig.sign(message)
            except TypeError:
                try:
                    with self._oqs.Signature(name, secret_key=signing_key) as sig:
                        return sig.sign(message)
                except Exception as e:
                    last_err = e
            except Exception as e:
                last_err = e
        raise ValueError(
            f"oqs: could not sign for {self.sig_id} (tried {self._name_candidates})."
        ) from last_err

    def verify(self, message: bytes, signature: bytes, verify_key: bytes) -> bool:
        """Verify a signature. Returns True if valid, False if invalid."""
        last_err = None
        for name in self._name_candidates:
            try:
                with self._new_sig() as sig:
                    return sig.verify(message, signature, verify_key)
            except Exception as e:
                last_err = e
        raise ValueError(
            f"oqs: could not verify for {self.sig_id} (tried {self._name_candidates})."
        ) from last_err


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
    """Generate X25519 + ML-KEM recipient key material for the hybrid scheme.

    Args:
        pq_kem_id: One of SUPPORTED_PQ_KEM_IDS (ML-KEM-512, ML-KEM-768, ML-KEM-1024).
    """
    validate_pq_kem_id(pq_kem_id)
    pq = OQSKEMBackend(pq_kem_id)
    pq_pk, pq_sk = pq.keygen()
    return HybridRecipientKeys(
        classical=generate_classical_recipient_keys(),
        pq_public_key=pq_pk,
        pq_secret_key=pq_sk,
        pq_kem_id=pq.kem_id,
    )


@dataclass(frozen=True)
class SigningKeys:
    """ML-DSA signing key bundle for envelope authentication.

    signing_key is the ML-DSA secret key (kept private by the signer).
    verify_key  is the ML-DSA public key (used by verifiers; not stored in DB).
    key_id      is a label used in records to identify which key was used.
    """
    sig_id:      str    # e.g. "ML-DSA-44"
    signing_key: bytes  # ML-DSA secret key
    verify_key:  bytes  # ML-DSA public key
    key_id:      str = "sig-key-v1"


def generate_sig_keys(
    sig_id: str = "ML-DSA-65",
    key_id: str = "sig-key-v1",
) -> SigningKeys:
    """Generate an ML-DSA signing key pair.

    Args:
        sig_id: One of SUPPORTED_PQ_SIG_IDS (ML-DSA-44, ML-DSA-65, ML-DSA-87).
        key_id: Label for key management / DB records.
    """
    validate_pq_sig_id(sig_id)
    backend = OQSSigBackend(sig_id)
    vk, sk = backend.keygen()
    return SigningKeys(sig_id=sig_id, signing_key=sk, verify_key=vk, key_id=key_id)


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

    # Signature fields (present only when ML-DSA envelope signing is enabled).
    # All three must be either all None (unsigned) or all set (signed).
    sig_alg_id: Optional[str] = None   # "ML-DSA-44", "ML-DSA-65", or "ML-DSA-87"
    signature:  Optional[bytes] = None # ML-DSA signature bytes over the canonical commitment
    sig_key_id: Optional[str] = None   # label identifying the signing key used

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


# -----------------------------
# Canonical envelope commitment
# -----------------------------

def _lv_encode(v: bytes) -> bytes:
    """Length-value encode: 4-byte big-endian length prefix followed by value bytes."""
    return struct.pack(">I", len(v)) + v


def _lv_str(s: Optional[str]) -> bytes:
    """Encode an optional string as UTF-8 length-value (empty bytes when None)."""
    return _lv_encode(b"" if s is None else s.encode("utf-8"))


def build_envelope_commitment(rec: EncryptedRecord) -> bytes:
    """Build a canonical, deterministic byte string committing to all
    security-relevant fields of an EncryptedRecord.

    This is the exact byte string that ML-DSA signs on the write path and
    that must be recomputed identically before verification on the read path.

    Fields covered (in order):
        domain_separator, version, scheme, payload_cipher, kem_id, kdf_id,
        wrap_id, ciphertext, nonce, tag, aad, wrapped_dek, ephemeral_pubkey,
        salt, hkdf_info, pq_kem_id, pq_ciphertext

    Encoding rules:
        - Domain separator: length-value encoded bytes.
        - version: 4-byte big-endian integer (no length prefix).
        - String fields: UTF-8 encoded, then length-value encoded.
          Optional string fields encode as empty bytes when None.
        - Bytes fields: length-value encoded.
          Optional bytes fields encode as empty bytes when None.

    The domain separator (COMMITMENT_DOMAIN_V1) prevents cross-context
    commitment confusion.
    """
    return b"".join([
        _lv_encode(COMMITMENT_DOMAIN_V1),
        struct.pack(">I", rec.version),
        _lv_str(rec.scheme),
        _lv_str(rec.payload_cipher),
        _lv_str(rec.kem_id),
        _lv_str(rec.kdf_id),
        _lv_str(rec.wrap_id),
        _lv_encode(rec.ciphertext),
        _lv_encode(rec.nonce),
        _lv_encode(rec.tag),
        _lv_encode(rec.aad),
        _lv_encode(rec.wrapped_dek),
        _lv_encode(rec.ephemeral_pubkey),
        _lv_encode(rec.salt),
        _lv_encode(rec.hkdf_info),
        _lv_str(rec.pq_kem_id),
        _lv_encode(rec.pq_ciphertext or b""),
    ])


def sign_envelope(
    rec: EncryptedRecord,
    signing_keys: SigningKeys,
    *,
    collect_metrics: bool = True,
) -> Tuple["EncryptedRecord", Dict[str, float]]:
    """Sign the canonical commitment of rec with ML-DSA.

    Returns a new EncryptedRecord (copy of rec with sig fields populated)
    and a metrics dict containing t_sig_sign_ms and signature_bytes.

    The record must already be encrypted before calling this function.
    Signing does not re-encrypt; it commits to the encrypted envelope fields.
    """
    backend = OQSSigBackend(signing_keys.sig_id)
    commitment = build_envelope_commitment(rec)

    metrics: Dict[str, float] = {}
    t0 = _now_ms()
    sig_bytes = backend.sign(commitment, signing_keys.signing_key)
    if collect_metrics:
        metrics["t_sig_sign_ms"] = _now_ms() - t0
        metrics["signature_bytes"] = float(len(sig_bytes))

    signed_rec = dataclasses.replace(
        rec,
        sig_alg_id=signing_keys.sig_id,
        signature=sig_bytes,
        sig_key_id=signing_keys.key_id,
    )
    return signed_rec, metrics


def verify_envelope(
    rec: EncryptedRecord,
    verify_key: bytes,
    *,
    collect_metrics: bool = True,
) -> Dict[str, float]:
    """Verify the ML-DSA signature on rec's canonical commitment.

    This must be called BEFORE decryption. Raises SigVerificationError if the
    signature does not verify. Returns a metrics dict with t_sig_verify_ms.

    Raises:
        ValueError: if the record has no signature fields (programmer error).
        SigVerificationError: if the signature is cryptographically invalid.
    """
    if rec.sig_alg_id is None or rec.signature is None:
        raise ValueError("Record has no signature to verify (sig_alg_id/signature are None)")

    backend = OQSSigBackend(rec.sig_alg_id)
    commitment = build_envelope_commitment(rec)

    metrics: Dict[str, float] = {}
    t0 = _now_ms()
    valid = backend.verify(commitment, rec.signature, verify_key)
    if collect_metrics:
        metrics["t_sig_verify_ms"] = _now_ms() - t0

    if not valid:
        raise SigVerificationError(
            f"ML-DSA signature verification failed (alg={rec.sig_alg_id!r}, "
            f"sig_key_id={rec.sig_key_id!r})"
        )
    return metrics


def envelope_sizes(rec: EncryptedRecord) -> Dict[str, int]:
    """Return the byte length of every stored field in an EncryptedRecord.

    Useful for storage-overhead analysis: compare sizes across schemes and
    signature modes to quantify bytes introduced by the PQ KEM / DSA layers.

    Definitions:
      header_bytes       = nonce + tag + wrapped_dek + eph_pubkey + salt + hkdf_info + aad
      overhead_bytes     = header_bytes + pq_ct_bytes + signature_bytes
      total_stored_bytes = ciphertext_bytes + overhead_bytes
    """
    pq_ct_len  = len(rec.pq_ciphertext) if rec.pq_ciphertext else 0
    sig_len    = len(rec.signature)      if rec.signature      else 0
    # "header" = fixed crypto metadata (excludes payload ciphertext, PQ ct, signature)
    header_bytes = (
        len(rec.nonce)
        + len(rec.tag)
        + len(rec.wrapped_dek)
        + len(rec.ephemeral_pubkey)
        + len(rec.salt)
        + len(rec.hkdf_info)
        + len(rec.aad)
    )
    overhead = header_bytes + pq_ct_len + sig_len
    total    = len(rec.ciphertext) + overhead
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
        "signature_bytes":     sig_len,
        "header_bytes":        header_bytes,
        "overhead_bytes":      overhead,
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

    # Signature fields: all-or-nothing consistency.
    sig_fields = (record.sig_alg_id, record.signature, record.sig_key_id)
    n_set = sum(1 for f in sig_fields if f is not None)
    if n_set not in (0, 3):
        raise ValueError(
            "Partial signature fields: sig_alg_id, signature, and sig_key_id "
            "must all be present (signed) or all absent (unsigned)"
        )
    if n_set == 3:
        if record.sig_alg_id not in SUPPORTED_PQ_SIG_IDS:
            raise ValueError(f"Unsupported sig_alg_id in record: {record.sig_alg_id!r}")


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
    validate_pq_kem_id(pq_kem_id)
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

    # hybrid test — exercise all three ML-KEM parameter sets
    rec_h = None
    hybrid_recipient = None
    for kem_id in SUPPORTED_PQ_KEM_IDS:
        hybrid_recipient = generate_hybrid_recipient_keys(kem_id)
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
        print(f"OK: hybrid encrypt/decrypt ({kem_id})")

    # tamper test (AEAD)
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

    # ── ML-DSA signature tests ────────────────────────────────────────────────
    print()
    print("== Stage 2 ML-DSA signature smoke test ==")

    for sig_id in SUPPORTED_PQ_SIG_IDS:
        sig_keys = generate_sig_keys(sig_id)
        # Use the last hybrid record from the loop above
        signed_rec, sign_metrics = sign_envelope(rec_h, sig_keys)
        assert signed_rec.sig_alg_id == sig_id
        assert signed_rec.signature is not None
        assert signed_rec.sig_key_id == sig_keys.key_id
        sig_size = len(signed_rec.signature)
        print(f"OK: sign_envelope ({sig_id})  signature={sig_size} B  t={sign_metrics.get('t_sig_sign_ms', 0):.3f} ms")

        # Verify the correct signature
        verify_metrics = verify_envelope(signed_rec, sig_keys.verify_key)
        print(f"OK: verify_envelope ({sig_id})  t={verify_metrics.get('t_sig_verify_ms', 0):.3f} ms")

        # Tamper: flip one bit of the ciphertext → commitment changes → verify fails
        tampered_ct = bytes([signed_rec.ciphertext[0] ^ 0x01]) + signed_rec.ciphertext[1:]
        tampered_signed = dataclasses.replace(signed_rec, ciphertext=tampered_ct)
        try:
            verify_envelope(tampered_signed, sig_keys.verify_key)
            raise AssertionError(f"Signature tamper NOT detected for {sig_id} (unexpected).")
        except SigVerificationError:
            print(f"OK: ciphertext tamper detected ({sig_id})")

        # Tamper: mutate AAD → commitment changes → verify fails
        tampered_aad = dataclasses.replace(signed_rec, aad=signed_rec.aad + b"X")
        try:
            verify_envelope(tampered_aad, sig_keys.verify_key)
            raise AssertionError(f"Metadata tamper NOT detected for {sig_id} (unexpected).")
        except SigVerificationError:
            print(f"OK: AAD tamper detected ({sig_id})")

    # Unsigned record has no sig fields
    assert rec_h.sig_alg_id is None
    assert rec_h.signature is None
    sizes = envelope_sizes(rec_h)
    assert sizes["signature_bytes"] == 0
    print("OK: unsigned record has zero signature_bytes in envelope_sizes")


if __name__ == "__main__":
    _smoke_test_stage2()
