"""Stable, deterministic digests for triton_runner cache keys.

Shared by the in-process JIT cache key (triton_runner/jit/versions.py) and
the on-disk cache key (triton_runner/compiler/compile.py). One normalizer
for both means a value digests identically in every process - set iteration
order (PYTHONHASHSEED) can never leak into a key - and any normalization fix
lands in both keys at once.
"""
import dataclasses
import hashlib
import json
import os


def _normalize_cache_key_value(value):
    if hasattr(value, "_asdict"):
        return _normalize_cache_key_value(value._asdict())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _normalize_cache_key_value(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {
            str(key): _normalize_cache_key_value(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_cache_key_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized_items = [_normalize_cache_key_value(item) for item in value]
        return sorted(
            normalized_items,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
        )
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, bytes):
        return {"__bytes__": value.hex()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "__dict__"):
        public_attrs = {
            key: attr
            for key, attr in vars(value).items()
            if not key.startswith("_")
        }
        if public_attrs:
            return _normalize_cache_key_value(public_attrs)
    return repr(value)


def stable_cache_key_digest(value):
    normalized = _normalize_cache_key_value(value)
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
