"""
app/elevation.py
================
Content-based ("selective security") elevation for FAERS safety reports.

Size-threshold policies elevate a record to Cat. 5 when its plaintext is
large.  Selective elevation instead flags important reports from their coded
content and elevates them regardless of size.  Every rule uses FAERS fields as
the FDA codes them, so it is reproducible from the public ASCII extracts.

Two principles, combinable:
  * clinical severity  - how bad the outcome was (OUTC codes)
  * special category   - data that privacy law singles out for extra
                         protection, independent of severity

OUTC (outcome) codes, per the FAERS ASCII README:
    DE Death   LT Life-threatening   HO Hospitalization   DS Disability
    CA Congenital anomaly   RI Required intervention   OT Other serious
  A report carries OUTC rows only when the outcome was serious, so
  "any OUTC code" is FAERS's own seriousness criterion.

Rules (approximate share of the 2x25,000 FAERS sample in brackets)
-----------------------------------------------------------------
Severity ladder (nested):
  death     DE                                        [~7%]   primary rule
  fatal_lt  DE or LT ("fatal or life-threatening": FDA's fastest, 7-day,
            IND safety-reporting category, 21 CFR 312.32(c)(2))  [~10%]
  severe    DE, LT, HO, DS, CA or RI  (serious, excluding "other serious"
            only)                                     [~28%]
  serious   any OUTC code                             [~55%]
Special-category components:
  pediatric   age < 18 y (age + age_cod) or age_grp N/I/C/T
  pregnancy   CA outcome, or a pregnancy / foetal / neonatal-exposure PT
  suicide     suicide or self-harm PT
  substance   substance-use-disorder PT (records protected by 42 CFR Part 2)
  hiv         HIV indication
  special     any of the five above                    [~10%]
Combined:
  sensitive       death or special                     [~16%]
  severe_special  severe or special                    [~34%]
Sensitivity variant:
  death_pt    DE, or a death PT in REAC (catches deaths coded only as a reaction)

Policy spec strings (used by app.paired_study and analyze_paired.py)
  flag:<rule>              elevate iff the rule matches
  flagsize:<rule>:<bytes>  elevate iff the rule matches OR size > bytes

Dependency-free, so the dataset builder (host) and the benchmark (container)
share one definition.
"""
from __future__ import annotations

import json
from typing import Dict, FrozenSet, Iterable, Optional

# ── MedDRA preferred-term lists (lower-case, exact match) ───────────────────
DEATH_PTS: FrozenSet[str] = frozenset({
    "death", "sudden death", "sudden cardiac death", "cardiac death", "completed suicide",
    "brain death", "accidental death", "death neonatal", "foetal death", "fetal death",
    "stillbirth", "apparent death",
})
SUBSTANCE_PTS: FrozenSet[str] = frozenset({
    "drug abuse", "substance abuse", "drug dependence", "substance dependence", "drug use disorder",
    "substance use disorder", "opioid use disorder", "alcohol use disorder", "alcoholism",
    "alcohol abuse", "dependence on opioids", "withdrawal syndrome", "drug withdrawal syndrome",
    "alcohol withdrawal syndrome", "drug withdrawal syndrome neonatal", "neonatal opioid withdrawal syndrome",
})
# substring keys (lower-case) for families too large to enumerate
SUICIDE_KEYS = ("suicid", "self-injurious", "self injurious", "intentional self-injury", "intentional overdose")
PREGNANCY_KEYS = ("pregnan", "foetal", "fetal", "abortion", "stillbirth", "premature baby",
                  "maternal exposure", "exposure during breast", "neonatal")
HIV_KEYS = ("hiv",)

SEVERE_CODES: FrozenSet[str] = frozenset({"DE", "LT", "HO", "DS", "CA", "RI"})
SERIOUS_CODES: FrozenSet[str] = SEVERE_CODES | {"OT"}
PEDIATRIC_AGE_GRP: FrozenSet[str] = frozenset({"N", "I", "C", "T"})
_AGE_UNIT_YEARS = {"YR": 1.0, "DEC": 10.0, "MON": 1 / 12, "WK": 1 / 52, "DY": 1 / 365, "HR": 1 / 8760}

SPECIAL_PARTS = ("pediatric", "pregnancy", "suicide", "substance", "hiv")
RULES = ("death", "death_pt", "fatal_lt", "severe", "serious",
         *SPECIAL_PARTS, "special", "sensitive", "severe_special")


def age_years(demo: dict) -> Optional[float]:
    try:
        a = float(demo.get("age", ""))
    except ValueError:
        return None
    f = _AGE_UNIT_YEARS.get(demo.get("age_cod", ""))
    return a * f if f is not None else None


def flags_from_doc(doc: dict) -> Dict[str, bool]:
    """Evaluate every rule on a parsed report."""
    oc = {o.strip().upper() for o in doc.get("outcomes", ()) if o}
    pts = [r.get("pt", "").strip().lower() for r in doc.get("reactions", ())]
    indis = [i.lower() for d in doc.get("drugs", ()) for i in d.get("indications", ())]
    demo = doc.get("demographics", {})
    age = age_years(demo)

    f = {
        "death": "DE" in oc,
        "fatal_lt": bool(oc & {"DE", "LT"}),
        "severe": bool(oc & SEVERE_CODES),
        "serious": bool(oc & SERIOUS_CODES),
        "pediatric": (age is not None and age < 18) or demo.get("age_grp") in PEDIATRIC_AGE_GRP,
        "pregnancy": "CA" in oc or any(k in p for p in pts for k in PREGNANCY_KEYS),
        "suicide": any(k in p for p in pts for k in SUICIDE_KEYS),
        "substance": any(p in SUBSTANCE_PTS for p in pts),
        "hiv": any(k in i for i in indis for k in HIV_KEYS),
    }
    f["death_pt"] = f["death"] or any(p in DEATH_PTS for p in pts)
    f["special"] = any(f[k] for k in SPECIAL_PARTS)
    f["sensitive"] = f["death"] or f["special"]
    f["severe_special"] = f["severe"] or f["special"]
    return f


def rule_matches(rule: str, plaintext: bytes) -> bool:
    """Classify a serialized report.  This is the work a policy engine does
    when it receives an opaque JSON document: parse it, then inspect the
    coded fields.  The harness times this call."""
    if rule not in RULES:
        raise ValueError(f"unknown flag rule {rule!r} (use one of {', '.join(RULES)})")
    doc = json.loads(plaintext)
    if rule == "death":  # fast path for the primary rule: no PT scan
        return "DE" in doc.get("outcomes", ())
    return flags_from_doc(doc)[rule]


def flags_from_fields(outcomes: Iterable[str], reaction_pts: Iterable[str]) -> Dict[str, bool]:
    """Backward-compatible helper: evaluate rules from outcome codes and PTs only."""
    return flags_from_doc({"outcomes": list(outcomes), "reactions": [{"pt": p} for p in reaction_pts]})
