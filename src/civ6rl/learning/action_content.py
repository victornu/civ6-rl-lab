"""Public content identities for three action fields missing from the actor vocabulary.

This module reads only an explicitly supplied, SHA-pinned gameplay SQLite
catalogue. It never reads a save, player state, map, or entity instance ID.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

SCHEMA = "civ6rl.learning.action-content-catalog.v1"
ROOT = Path(__file__).resolve().parents[2]
GAMEINFO_EVIDENCE_PATH = ROOT / 'evidence/phase41-gameinfo-maps.json'
GAMEINFO_EVIDENCE_SHA256 = '61f762c40ab5a1f3789d4afdca711042d19083390ff800203bc3e091d8606a45'
TABLE_KEYS = {
    "Units": "UnitType",
    "GreatPersonIndividuals": "GreatPersonIndividualType",
    "Beliefs": "BeliefType",
}
TYPE_RE = re.compile(r"[A-Z][A-Z0-9_]{0,127}\Z")
BELIEF_CLASS_RE = re.compile(r"BELIEF_CLASS_[A-Z0-9_]{1,120}\Z")
MAX_DATABASE_BYTES = 1_073_741_824
MAX_ROWS_PER_TABLE = 100_000
MAX_TYPE_ROWS = 500_000
DATABASE_SIDECARS = ("-wal", "-shm", "-journal")


class ActionContentCatalogError(ValueError):
    """The explicit public database does not satisfy its declared contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quoted(name: str) -> str:
    if type(name) is not str or not name or "\x00" in name:
        raise ActionContentCatalogError("invalid SQLite identifier")
    return '"' + name.replace('"', '""') + '"'


def _validate_sha(value: str) -> str:
    if (type(value) is not str or len(value) != 64
            or any(ch not in "0123456789abcdef" for ch in value)):
        raise ActionContentCatalogError("lowercase expected_sha256 is required")
    return value


def _require_no_sidecars(db_path: Path) -> None:
    if any(Path(str(db_path) + suffix).exists() for suffix in DATABASE_SIDECARS):
        raise ActionContentCatalogError("public gameplay database has mutable SQLite sidecars")


def extract_action_content_catalog(path: str | Path, *, expected_sha256: str) -> dict[str, Any]:
    """Extract public GameInfo identities from an explicit, SHA-pinned DB.

    GameInfo indices are zero-based ordinals from ``ORDER BY rowid``. The
    database is opened read-only; only the three named public content tables
    and ``Types`` are read.
    """
    expected_sha256 = _validate_sha(expected_sha256)
    db_path = Path(path).resolve(strict=True)
    if not db_path.is_file() or db_path.stat().st_size > MAX_DATABASE_BYTES:
        raise ActionContentCatalogError("public gameplay database missing or over size bound")
    _require_no_sidecars(db_path)
    evidence_before = _sha256(GAMEINFO_EVIDENCE_PATH)
    if evidence_before != GAMEINFO_EVIDENCE_SHA256:
        raise ActionContentCatalogError("pinned GameInfo ordering evidence changed")
    before = _sha256(db_path)
    if before != expected_sha256:
        raise ActionContentCatalogError("public gameplay database SHA-256 differs")

    result: dict[str, Any] = {"schema": SCHEMA, "database_sha256": before, "gameinfo_evidence_sha256": evidence_before, "tables": {}}
    try:
        db = sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True)
        try:
            db.execute("PRAGMA query_only=ON")
            if db.execute("PRAGMA quick_check(1)").fetchone() != ("ok",):
                raise ActionContentCatalogError("public gameplay database quick_check failed")
            type_info = db.execute('PRAGMA table_info("Types")').fetchall()
            if not type_info or [r[1] for r in type_info if r[5]] != ["Type"]:
                raise ActionContentCatalogError("Types primary key differs from contract")
            if not {"Type", "Hash"} <= {r[1] for r in type_info}:
                raise ActionContentCatalogError("Types lacks Type or Hash")
            type_count = db.execute('SELECT count(*) FROM "Types"').fetchone()[0]
            if type(type_count) is not int or not 0 < type_count <= MAX_TYPE_ROWS:
                raise ActionContentCatalogError("Types row count outside its bound")
            hashes = {}
            for type_name, raw_hash in db.execute('SELECT "Type","Hash" FROM "Types"'):
                if (type(type_name) is not str or TYPE_RE.fullmatch(type_name) is None
                        or type(raw_hash) is not int
                        or not -(1 << 31) <= raw_hash <= 0xFFFFFFFF or type_name in hashes):
                    raise ActionContentCatalogError("invalid or duplicate Types row")
                hashes[type_name] = raw_hash & 0xFFFFFFFF

            if len(hashes) != type_count:
                raise ActionContentCatalogError("Types row count changed while reading")

            for table, key in TABLE_KEYS.items():
                info = db.execute("PRAGMA table_info(" + _quoted(table) + ")").fetchall()
                columns = {r[1] for r in info}
                if not info or [r[1] for r in info if r[5]] != [key] or key not in columns:
                    raise ActionContentCatalogError(f"{table} primary key differs from contract")
                if table == "Beliefs" and "BeliefClassType" not in columns:
                    raise ActionContentCatalogError("Beliefs lacks BeliefClassType")
                row_count = db.execute("SELECT count(*) FROM " + _quoted(table)).fetchone()[0]
                if type(row_count) is not int or row_count > MAX_ROWS_PER_TABLE:
                    raise ActionContentCatalogError(f"{table} exceeds row bound")
                select = 'SELECT rowid,' + _quoted(key)
                if table == "Beliefs":
                    select += ',"BeliefClassType"'
                rows = db.execute(select + " FROM " + _quoted(table) + " ORDER BY rowid")
                entries = []
                seen_types = set()
                for index, row in enumerate(rows):
                    row_id, type_name = row[:2]
                    if (type(row_id) is not int or type(type_name) is not str
                            or TYPE_RE.fullmatch(type_name) is None or type_name in seen_types
                            or type_name not in hashes):
                        raise ActionContentCatalogError(f"invalid {table} public identity")
                    seen_types.add(type_name)
                    entry = {"index": index, "type": type_name, "hash": hashes[type_name]}
                    if table == "Beliefs":
                        belief_class = row[2]
                        if (type(belief_class) is not str
                                or BELIEF_CLASS_RE.fullmatch(belief_class) is None):
                            raise ActionContentCatalogError("invalid Beliefs.BeliefClassType")
                        entry["class"] = belief_class
                    entries.append(entry)
                result["tables"][table] = entries
        finally:
            db.close()
    except sqlite3.Error as exc:
        raise ActionContentCatalogError("invalid or unreadable gameplay SQLite schema") from exc
    if _sha256(db_path) != before:
        raise ActionContentCatalogError("public gameplay database changed during extraction")
    _require_no_sidecars(db_path)
    if _sha256(GAMEINFO_EVIDENCE_PATH) != evidence_before:
        raise ActionContentCatalogError("GameInfo ordering evidence changed during extraction")
    return result


def project_action_content(action: dict[str, Any], catalog: dict[str, Any]) -> dict[str, Any]:
    """Purely join action identities to existing categorical vocabulary tokens.

    Returned values include the joined public identity and categorical token.
    This projection does not establish native offer availability; callers
    must separately join against the current native action catalogue.
    """
    if type(catalog) is not dict or catalog.get("schema") != SCHEMA:
        raise ActionContentCatalogError("unsupported action content catalogue")
    tables = catalog.get("tables")
    if type(tables) is not dict or set(tables) != set(TABLE_KEYS):
        raise ActionContentCatalogError("catalogue table set differs from contract")
    if type(action) is not dict or type(action.get("kind")) is not str:
        raise ActionContentCatalogError("malformed action")

    kind = action["kind"]
    if kind == "upgrade_unit":
        field, table, namespace = "target_type", "Units", "action.target_type"
    elif kind == "recruit_great_person":
        field, table, namespace = "individual", "GreatPersonIndividuals", "action.individual"
    elif kind == "add_belief":
        field, table, namespace = "class", "Beliefs", "action.class"
    else:
        return {}
    if field not in action:
        raise ActionContentCatalogError(f"{kind} lacks {field}")
    entries = tables[table]
    if type(entries) is not list:
        raise ActionContentCatalogError(f"malformed {table} catalogue")
    if kind == "add_belief":
        value = action[field]
        if type(value) is not str or BELIEF_CLASS_RE.fullmatch(value) is None:
            raise ActionContentCatalogError("add_belief.class is not a native BELIEF_CLASS value")
        index, hash_value = action.get("index"), action.get("hash")
        if (type(index) is not int or index < 0 or type(hash_value) is not int
                or not 0 <= hash_value <= 0xFFFFFFFF):
            raise ActionContentCatalogError("add_belief requires native index and hash")
        matches = [row for row in entries if type(row) is dict
                   and row.get("index") == index and row.get("hash") == hash_value]
        if len(matches) != 1 or matches[0].get("class") != value:
            raise ActionContentCatalogError("belief index/hash/class do not identify one public Beliefs row")
        row = matches[0]
        return {field: {"table": table, "index": row["index"], "type": row["type"],
                        "hash": row["hash"], "class": value,
                        "token": namespace + ":" + json.dumps(value, ensure_ascii=True,
                                                                  separators=(",", ":"))}}
    value = action[field]
    if type(value) is not int or value < 0:
        raise ActionContentCatalogError(f"{kind}.{field} must be a nonnegative GameInfo index")
    matches = [row for row in entries if type(row) is dict and row.get("index") == value]
    if len(matches) != 1:
        raise ActionContentCatalogError(f"{kind}.{field} is absent or ambiguous in public catalogue")
    row = matches[0]
    return {field: {"table": table, "index": row["index"], "type": row["type"],
                    "hash": row["hash"], "token": namespace + ":" + str(value)}}


__all__ = ["SCHEMA", "TABLE_KEYS", "ActionContentCatalogError",
           "extract_action_content_catalog", "project_action_content"]





