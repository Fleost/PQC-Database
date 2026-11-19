"""Quick correctness checks for the classical baseline and flag policies.

    docker compose run --rm benchmark python /app/scripts/check_classical.py
"""
import dataclasses
import json
import sys

sys.path.insert(0, "/app")
from app import elevation, stage2  # noqa: E402
from app.paired_study import resolve_profile  # noqa: E402

pt = b'{"outcomes": ["DE", "OT"], "reactions": [{"pt": "Sepsis"}]}'
keys = stage2.generate_classical_recipient_keys()
sk = stage2.generate_sig_keys(stage2.SIG_ID_ECDSA_P256)
rec = stage2.encrypt(pt, scheme="classical", recipient_classical_pk=keys.pk, aad=b"x")
rec, m = stage2.sign_envelope(rec, sk)
sizes = stage2.envelope_sizes(rec)
stage2.verify_envelope(rec, sk.verify_key)
out, _ = stage2.decrypt(rec, recipient_classical_sk=keys.sk)
assert out == pt
print(f"OK classical round-trip: sig={sizes['signature_bytes']} B pq_ct={sizes['pq_ct_bytes']} B "
      f"overhead={sizes['overhead_bytes']} B")
bad = dataclasses.replace(rec, ciphertext=bytes([rec.ciphertext[0] ^ 1]) + rec.ciphertext[1:])
try:
    stage2.verify_envelope(bad, sk.verify_key)
    raise SystemExit("FAIL: ECDSA tamper not detected")
except stage2.SigVerificationError:
    print("OK ECDSA tamper detected")
other = stage2.generate_sig_keys(stage2.SIG_ID_ECDSA_P256)
try:
    stage2.verify_envelope(rec, other.verify_key)
    raise SystemExit("FAIL: wrong key accepted")
except stage2.SigVerificationError:
    print("OK ECDSA wrong key rejected")

cases = [
    ("flag:death", pt, 100, "strong"),
    ("flag:death", b'{"outcomes": ["HO"], "reactions": []}', 100, "baseline"),
    ("flag:death_pt", b'{"outcomes": [], "reactions": [{"pt": "Completed suicide"}]}', 100, "strong"),
    ("flag:fatal_lt", b'{"outcomes": ["LT"], "reactions": []}', 100, "strong"),
    ("flag:serious", b'{"outcomes": [], "reactions": []}', 100, "baseline"),
    ("flagsize:death:4196", b'{"outcomes": [], "reactions": []}', 5000, "strong"),
    ("flagsize:death:4196", b'{"outcomes": [], "reactions": []}', 100, "baseline"),
    ("flag:severe", b'{"outcomes": ["OT"], "reactions": []}', 100, "baseline"),
    ("flag:severe", b'{"outcomes": ["HO", "OT"], "reactions": []}', 100, "strong"),
    ("flag:special", b'{"outcomes": [], "reactions": [], "demographics": {"age": "7", "age_cod": "YR"}}', 100, "strong"),
    ("flag:special", b'{"outcomes": [], "reactions": [], "demographics": {"age": "40", "age_cod": "YR"}}', 100, "baseline"),
    ("flag:special", b'{"outcomes": [], "reactions": [{"pt": "Loss of personal independence in daily activities"}]}', 100, "baseline"),
    ("flag:special", b'{"outcomes": [], "reactions": [{"pt": "Drug dependence"}]}', 100, "strong"),
    ("flag:sensitive", b'{"outcomes": [], "reactions": [], "drugs": [{"indications": ["HIV infection"]}]}', 100, "strong"),
    ("adaptive:1458", b"{}", 1459, "strong"),
    ("adaptive:1458", b"{}", 1458, "baseline"),
    ("classical", b"{}", 1, "classical"),
]
for spec, data, size, want in cases:
    got = resolve_profile(spec, size, data)[2]
    assert got == want, (spec, size, got, want)
print(f"OK {len(cases)} policy decisions")
