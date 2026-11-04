"""Bounded stdlib JSON-lines codec for learning worker transport.

Messages use compact finite JSON while they fit one protocol line. Larger
objects use a single explicit zlib/base64 envelope with an uncompressed size
and SHA-256. The reserved marker may not appear in application messages.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import zlib

# A bounded recurrent collection may exceed a 1 MiB compressed envelope.
# Legacy defaults stay intact. Larger pairs require a negotiated episode budget.
MAX_LINE_BYTES = 2 * 1024 * 1024
MAX_DECODED_BYTES = 8 * 1024 * 1024
EXTENDED_MAX_DECODED_BYTES = 384 * 1024 * 1024
# A plain JSON message can be at most the decoded limit, plus its newline.
# This also bounds envelopes while allowing any permitted message to fit even
# when compression makes it larger.
EXTENDED_MAX_LINE_BYTES = EXTENDED_MAX_DECODED_BYTES + 1
WIRE_MARKER = '$civ6rl_wire'
WIRE_VERSION = 1
_SHA256 = re.compile(r'[0-9a-f]{64}\Z')
_ENVELOPE_FIELDS = {'version', 'format', 'raw_bytes', 'sha256', 'data'}


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError('non-finite JSON number')
    return value


def _reject_constant(text: str):
    raise ValueError(f'non-finite JSON constant: {text}')


def _parse_json(raw: bytes) -> dict:
    try:
        text = raw.decode('utf-8', errors='strict')
    except UnicodeDecodeError as exc:
        raise ValueError('wire JSON is not valid UTF-8') from exc
    try:
        value = json.loads(text, object_pairs_hook=_pairs,
                           parse_float=_finite_float, parse_constant=_reject_constant)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValueError('wire payload is malformed JSON') from exc
    if type(value) is not dict:
        raise ValueError('wire payload root must be an object')
    return value


def _contains_marker(value) -> bool:
    pending = [value]
    visited = set()
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            identity = id(item)
            if identity in visited:
                continue
            visited.add(identity)
            if WIRE_MARKER in item:
                return True
            if any(type(key) is not str for key in item):
                raise ValueError('wire JSON object keys must be strings')
            pending.extend(item.values())
        elif isinstance(item, (list, tuple)):
            identity = id(item)
            if identity in visited:
                continue
            visited.add(identity)
            pending.extend(item)
    return False


def _limits(max_line_bytes, max_decoded_bytes):
    if type(max_line_bytes) is not int or type(max_decoded_bytes) is not int:
        raise ValueError('wire limits must be the bounded default or extended pair')
    pair = (max_line_bytes, max_decoded_bytes)
    if pair not in ((MAX_LINE_BYTES, MAX_DECODED_BYTES),
                    (EXTENDED_MAX_LINE_BYTES, EXTENDED_MAX_DECODED_BYTES)):
        raise ValueError('wire limits must be the bounded default or extended pair')
    return pair


def _dump(value: dict, max_decoded_bytes=MAX_DECODED_BYTES) -> bytes:
    if type(value) is not dict:
        raise ValueError('wire message root must be an object')
    if _contains_marker(value):
        raise ValueError('reserved wire marker in application message')
    try:
        raw = json.dumps(value, ensure_ascii=False, allow_nan=False,
                         separators=(',', ':')).encode('utf-8', errors='strict')
    except (TypeError, ValueError, UnicodeEncodeError, RecursionError) as exc:
        raise ValueError('wire message is not finite JSON') from exc
    if len(raw) > max_decoded_bytes:
        raise ValueError(f'decoded wire message exceeds {max_decoded_bytes // (1024 * 1024)} MiB limit')
    return raw


def encode_line(message: dict, *, max_line_bytes=MAX_LINE_BYTES,
                max_decoded_bytes=MAX_DECODED_BYTES) -> bytes:
    """Encode one dict as a newline-terminated bounded protocol frame."""
    max_line_bytes, max_decoded_bytes = _limits(max_line_bytes, max_decoded_bytes)
    raw = _dump(message, max_decoded_bytes)
    plain = raw + b'\n'
    # Keep the legacy framing behavior for small messages. Once a message
    # crosses that threshold, compression is worthwhile to try even when an
    # extended line budget can carry the raw JSON.
    if len(plain) <= MAX_LINE_BYTES:
        return plain
    compressed = zlib.compress(raw)
    envelope = {
        WIRE_MARKER: {
            'version': WIRE_VERSION,
            'format': 'zlib+base64',
            'raw_bytes': len(raw),
            'sha256': hashlib.sha256(raw).hexdigest(),
            'data': base64.b64encode(compressed).decode('ascii'),
        }
    }
    try:
        line = json.dumps(envelope, ensure_ascii=True, allow_nan=False,
                          separators=(',', ':')).encode('ascii') + b'\n'
    except (TypeError, ValueError) as exc:  # pragma: no cover - fixed envelope
        raise ValueError('could not encode wire envelope') from exc
    if len(line) < len(plain) and len(line) <= max_line_bytes:
        return line
    if len(plain) <= max_line_bytes:
        return plain
    raise ValueError('wire envelope exceeds line limit')


def _decode_envelope(wrapper: dict, max_decoded_bytes=MAX_DECODED_BYTES) -> dict:
    if set(wrapper) != {WIRE_MARKER}:
        raise ValueError('unknown or extra wire envelope fields')
    envelope = wrapper[WIRE_MARKER]
    if type(envelope) is not dict or set(envelope) != _ENVELOPE_FIELDS:
        raise ValueError('invalid wire envelope fields')
    if type(envelope['version']) is not int or envelope['version'] != WIRE_VERSION:
        raise ValueError('unsupported wire envelope version')
    if envelope['format'] != 'zlib+base64':
        raise ValueError('unsupported wire envelope format')
    expected_size = envelope['raw_bytes']
    if (type(expected_size) is not int or expected_size < 0
            or expected_size > max_decoded_bytes):
        raise ValueError('invalid wire decoded size')
    expected_hash = envelope['sha256']
    if type(expected_hash) is not str or not _SHA256.fullmatch(expected_hash):
        raise ValueError('invalid wire SHA-256')
    encoded = envelope['data']
    if type(encoded) is not str:
        raise ValueError('invalid wire base64 data')
    try:
        compressed = base64.b64decode(encoded.encode('ascii'), validate=True)
    except (UnicodeEncodeError, ValueError) as exc:
        raise ValueError('invalid wire base64 data') from exc
    inflater = zlib.decompressobj()
    try:
        raw = inflater.decompress(compressed, max_decoded_bytes + 1)
    except zlib.error as exc:
        raise ValueError('invalid compressed wire payload') from exc
    if len(raw) > max_decoded_bytes:
        raise ValueError(f'decoded wire message exceeds {max_decoded_bytes // (1024 * 1024)} MiB limit')
    if not inflater.eof:
        # No flush: a missing stream terminator or output beyond the bounded
        # decompression window is rejected without unbounded expansion.
        raise ValueError('truncated or oversized compressed wire payload')
    if inflater.unused_data or inflater.unconsumed_tail:
        raise ValueError('extra data after compressed wire stream')
    if len(raw) != expected_size:
        raise ValueError('wire decoded size mismatch')
    if hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValueError('wire SHA-256 mismatch')
    value = _parse_json(raw)
    if _contains_marker(value):
        raise ValueError('nested reserved wire envelope')
    return value


def decode_line(line: bytes, *, max_line_bytes=MAX_LINE_BYTES,
                max_decoded_bytes=MAX_DECODED_BYTES) -> dict:
    """Decode exactly one bounded newline-terminated JSON protocol frame."""
    max_line_bytes, max_decoded_bytes = _limits(max_line_bytes, max_decoded_bytes)
    if type(line) is not bytes:
        raise ValueError('wire line must be bytes')
    if len(line) > max_line_bytes:
        raise ValueError('wire line exceeds limit')
    if not line.endswith(b'\n') or line.count(b'\n') != 1:
        raise ValueError('wire frame must contain exactly one complete line')
    raw = line[:-1]
    if not raw:
        raise ValueError('wire frame is empty')
    value = _parse_json(raw)
    if WIRE_MARKER in value:
        return _decode_envelope(value, max_decoded_bytes)
    if _contains_marker(value):
        raise ValueError('reserved wire marker in application message')
    return value


__all__ = ['MAX_LINE_BYTES', 'MAX_DECODED_BYTES', 'EXTENDED_MAX_LINE_BYTES',
           'EXTENDED_MAX_DECODED_BYTES', 'WIRE_MARKER',
           'encode_line', 'decode_line']
