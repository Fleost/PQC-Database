"""
Workload-aware PQ parameter assignment policy.

Selects between a baseline and a strong PQ profile based on record size.
This module is a pure policy layer — it does not perform any cryptographic
operations and does not depend on the database or service layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLD_BYTES: int = 50 * 1024  # 50 KiB

BASELINE_KEM_ID: str = "ML-KEM-768"
BASELINE_SIG_ID: str = "ML-DSA-65"

STRONG_KEM_ID: str = "ML-KEM-1024"
STRONG_SIG_ID: str = "ML-DSA-87"


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class ThresholdPolicyConfig:
    """Configuration for the size-threshold PQ parameter policy.

    Records strictly larger than ``threshold_bytes`` receive the strong
    profile; all others receive the baseline profile.
    """

    threshold_bytes: int = DEFAULT_THRESHOLD_BYTES
    baseline_kem_id: str = BASELINE_KEM_ID
    baseline_sig_id: str = BASELINE_SIG_ID
    strong_kem_id: str = STRONG_KEM_ID
    strong_sig_id: str = STRONG_SIG_ID


# ---------------------------------------------------------------------------
# Decision dataclass
# ---------------------------------------------------------------------------


@dataclass
class PolicyDecision:
    """The result of a policy evaluation for a single record.

    Attributes:
        record_size_bytes: The size of the record that was evaluated.
        threshold_bytes:   The threshold used during evaluation.
        kem_id:            Selected KEM algorithm identifier.
        sig_id:            Selected signature algorithm identifier.
        tier_label:        Human-readable tier name ("baseline" or "strong").
        escalated:         True when the strong profile was selected.
    """

    record_size_bytes: int
    threshold_bytes: int
    kem_id: str
    sig_id: str
    tier_label: str
    escalated: bool


# ---------------------------------------------------------------------------
# Public selection function
# ---------------------------------------------------------------------------


def select_policy_for_record(
    record_size_bytes: int,
    config: ThresholdPolicyConfig | None = None,
) -> PolicyDecision:
    """Return the appropriate PQ policy decision for a record of the given size.

    The rule is strictly greater-than:
      * ``record_size_bytes > threshold_bytes``  → strong profile
      * ``record_size_bytes <= threshold_bytes`` → baseline profile

    Args:
        record_size_bytes: The byte size of the plaintext record.
        config: Optional policy configuration; defaults to
                :class:`ThresholdPolicyConfig` with its default values.

    Returns:
        A :class:`PolicyDecision` describing the selected profile.

    Raises:
        TypeError:  If ``record_size_bytes`` is not an ``int``.
        ValueError: If ``record_size_bytes`` is negative.
        TypeError:  If ``config.threshold_bytes`` is not an ``int``.
        ValueError: If ``config.threshold_bytes`` is negative.
    """
    if config is None:
        config = ThresholdPolicyConfig()

    _validate_record_size(record_size_bytes)
    _validate_threshold(config.threshold_bytes)

    escalated = record_size_bytes > config.threshold_bytes

    if escalated:
        kem_id = config.strong_kem_id
        sig_id = config.strong_sig_id
        tier_label = "strong"
    else:
        kem_id = config.baseline_kem_id
        sig_id = config.baseline_sig_id
        tier_label = "baseline"

    return PolicyDecision(
        record_size_bytes=record_size_bytes,
        threshold_bytes=config.threshold_bytes,
        kem_id=kem_id,
        sig_id=sig_id,
        tier_label=tier_label,
        escalated=escalated,
    )


# ---------------------------------------------------------------------------
# Optional convenience helpers
# ---------------------------------------------------------------------------


def is_strong_assignment(
    record_size_bytes: int,
    config: ThresholdPolicyConfig | None = None,
) -> bool:
    """Return True if the record would receive the strong PQ profile."""
    return select_policy_for_record(record_size_bytes, config).escalated


def format_threshold_label(threshold_bytes: int) -> str:
    """Return a human-readable label for a threshold value.

    Examples:
        10485760 → "10.0 MiB"
        1048576  → "1.0 MiB"
        512      → "512 B"
    """
    if not isinstance(threshold_bytes, int):
        raise TypeError(
            f"threshold_bytes must be an int, got {type(threshold_bytes).__name__!r}"
        )
    if threshold_bytes < 0:
        raise ValueError(
            f"threshold_bytes must be >= 0, got {threshold_bytes}"
        )
    mib = threshold_bytes / (1024 * 1024)
    if mib >= 1.0:
        return f"{mib:.1f} MiB"
    kib = threshold_bytes / 1024
    if kib >= 1.0:
        return f"{kib:.1f} KiB"
    return f"{threshold_bytes} B"


# ---------------------------------------------------------------------------
# Internal validators
# ---------------------------------------------------------------------------


def _validate_record_size(record_size_bytes: int) -> None:
    if not isinstance(record_size_bytes, int):
        raise TypeError(
            f"record_size_bytes must be an int, got {type(record_size_bytes).__name__!r}"
        )
    if record_size_bytes < 0:
        raise ValueError(
            f"record_size_bytes must be >= 0, got {record_size_bytes}"
        )


def _validate_threshold(threshold_bytes: int) -> None:
    if not isinstance(threshold_bytes, int):
        raise TypeError(
            f"threshold_bytes must be an int, got {type(threshold_bytes).__name__!r}"
        )
    if threshold_bytes < 0:
        raise ValueError(
            f"threshold_bytes must be >= 0, got {threshold_bytes}"
        )
