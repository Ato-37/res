"""Verify exact scorer kwargs at the final PyTorch ``forward`` entry.

This is a software-path integrity check, not model/runtime authentication or a
claim-faithfulness result. Wrapping the forward method verifies final args and
kwargs after PyTorch global and module pre-hooks, immediately before the
original model forward body.
"""
from __future__ import annotations

from typing import Any, Callable

from nextgqa_atomic_batch_freeze import batch_sha256


MODEL_OPTIONS = {"use_cache": False, "logits_to_keep": 1}


class ModelCallBoundary:
    """Fail-closed wrapper with aggregate counters for exact root-call checks."""

    def __init__(self, *, on_root_entry: Callable[[], None] | None = None):
        self.root_call_entries = 0
        self.verified_root_calls = 0
        self.completed_calls = 0
        self.on_root_entry = on_root_entry

    @staticmethod
    def _validate_kwargs(call_kwargs: Any, *, expected_batch_sha256: str,
                         batch_keys: tuple[str, ...]) -> None:
        if type(call_kwargs) is not dict:
            raise ValueError("model_call_kwargs_must_be_plain_dict")
        if (not isinstance(batch_keys, tuple)
                or any(type(key) is not str or not key for key in batch_keys)
                or len(set(batch_keys)) != len(batch_keys)
                or set(batch_keys).intersection(MODEL_OPTIONS)):
            raise ValueError("model_call_batch_key_schema_invalid")
        if set(call_kwargs) != set(batch_keys).union(MODEL_OPTIONS):
            raise ValueError("model_call_kwargs_schema_mismatch")
        if (call_kwargs.get("use_cache") is not False
                or type(call_kwargs.get("logits_to_keep")) is not int
                or call_kwargs.get("logits_to_keep") != 1):
            raise ValueError("model_call_options_mismatch")
        if (not isinstance(expected_batch_sha256, str)
                or len(expected_batch_sha256) != 64
                or any(char not in "0123456789abcdef" for char in expected_batch_sha256)):
            raise ValueError("model_call_expected_digest_invalid")
        try:
            observed = batch_sha256({key: call_kwargs[key] for key in batch_keys})
        except Exception:
            raise ValueError("model_call_batch_digest_unavailable") from None
        if observed != expected_batch_sha256:
            raise ValueError("model_call_batch_digest_mismatch")

    def __call__(self, *, model: Any, call_kwargs: dict[str, Any],
                 expected_batch_keys: tuple[str, ...],
                 expected_batch_sha256: str, row_position: int,
                 condition_position: int, condition: str) -> Any:
        """Call one row-condition and verify kwargs at the forward entry."""
        if (model is None or type(row_position) is not int or row_position < 1
                or type(condition_position) is not int or condition_position < 1
                or not isinstance(condition, str) or not condition):
            raise ValueError("model_call_identity_invalid")
        if type(call_kwargs) is not dict:
            raise ValueError("model_call_kwargs_must_be_plain_dict")
        batch_keys = expected_batch_keys
        if (not isinstance(batch_keys, tuple)
                or any(type(key) is not str or not key for key in batch_keys)
                or len(set(batch_keys)) != len(batch_keys)
                or set(batch_keys).intersection(MODEL_OPTIONS)):
            raise ValueError("model_call_batch_key_schema_invalid")
        if set(call_kwargs) != set(batch_keys).union(MODEL_OPTIONS):
            raise ValueError("model_call_kwargs_schema_mismatch")

        entries_this_call = 0

        original_forward = getattr(model, "forward", None)
        if not callable(original_forward):
            raise ValueError("model_call_forward_unavailable")
        model_dict = getattr(model, "__dict__", {})
        had_instance_forward = "forward" in model_dict
        previous_instance_forward = model_dict.get("forward")

        def verify_forward_entry(*args, **kwargs):
            nonlocal entries_this_call
            self.root_call_entries += 1
            entries_this_call += 1
            if self.on_root_entry is not None:
                self.on_root_entry()
            if args:
                raise ValueError("model_call_forward_positional_args_mismatch")
            self._validate_kwargs(
                kwargs, expected_batch_sha256=expected_batch_sha256,
                batch_keys=batch_keys)
            self.verified_root_calls += 1

            return original_forward(*args, **kwargs)

        try:
            # torch.nn.Module captures ``self.forward`` before dispatching its
            # pre-hooks, then invokes this wrapper with their final args/kwargs.
            model.forward = verify_forward_entry
            output = model(**call_kwargs)
        finally:
            try:
                if had_instance_forward:
                    model.forward = previous_instance_forward
                else:
                    delattr(model, "forward")
            except Exception:
                raise ValueError("model_call_forward_wrapper_cleanup_failed") from None
        if entries_this_call != 1:
            raise ValueError("model_call_forward_entry_count_mismatch")
        self.completed_calls += 1
        return output
