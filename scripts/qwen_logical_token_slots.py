"""Map stable full-sequence token slots into append-only cached query slots.

This is a pure coordinate helper, not a model hook. It applies only to an
unpadded, unbatched append-only cache whose actual length matches the caller's
expected length and query origin. Callers should skip a hook call when the
logical target is outside the current query window; an out-of-window error
means the requested intervention cannot be applied on this call.

Static/sliding caches, padding, batching, beams, and model integration are not
covered. A local ``sequence_index=-1`` is not a stable logical token identity.
"""

from collections.abc import MutableSequence
from typing import TypeVar


T = TypeVar("T")


def _require_int(name: str, value: object) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an int (bool is not accepted)")
    return value


def logical_slot_to_local(
    logical_slot: int,
    *,
    query_origin: int,
    query_length: int,
    actual_cache_length: int,
    expected_cache_length: int,
) -> int:
    """Return a logical full-sequence slot's index in this decoder query.

    The supported append-only contract is ``query_origin == expected cache
    length == actual cache length``. The logical slot must be present in the
    current query interval ``[query_origin, query_origin + query_length)``.
    """
    logical_slot = _require_int("logical_slot", logical_slot)
    query_origin = _require_int("query_origin", query_origin)
    query_length = _require_int("query_length", query_length)
    actual_cache_length = _require_int("actual_cache_length", actual_cache_length)
    expected_cache_length = _require_int("expected_cache_length", expected_cache_length)

    if logical_slot < 0:
        raise ValueError("logical_slot must be nonnegative")
    if query_origin < 0:
        raise ValueError("query_origin must be nonnegative")
    if query_length <= 0:
        raise ValueError("query_length must be positive")
    if actual_cache_length < 0 or expected_cache_length < 0:
        raise ValueError("cache lengths must be nonnegative")
    if actual_cache_length != expected_cache_length:
        raise ValueError("actual cache length differs from expected cache length")
    if query_origin != expected_cache_length:
        raise ValueError("query origin must equal expected cache length for append-only decoding")

    local_slot = logical_slot - query_origin
    if local_slot < 0 or local_slot >= query_length:
        raise ValueError("logical slot is outside this query window; caller must skip this call")
    return local_slot


def replace_logical_slot(
    sequence: MutableSequence[T],
    donor_value: T,
    *,
    logical_slot: int,
    query_origin: int,
    query_length: int,
    actual_cache_length: int,
    expected_cache_length: int,
) -> int:
    """Replace one current-query element after all coordinates validate.

    The small mutable-sequence API makes the validation-before-mutation
    contract testable without importing a model framework. Model hooks should
    compute the returned local slot first, then patch their tensor only after
    all shape/device/dtype checks also pass.
    """
    query_length = _require_int("query_length", query_length)
    if not isinstance(sequence, MutableSequence):
        raise TypeError("sequence must be a mutable sequence")
    if len(sequence) != query_length:
        raise ValueError("sequence length must equal query_length")
    local_slot = logical_slot_to_local(
        logical_slot,
        query_origin=query_origin,
        query_length=query_length,
        actual_cache_length=actual_cache_length,
        expected_cache_length=expected_cache_length,
    )
    sequence[local_slot] = donor_value
    return local_slot
