"""Build a frozen public-content vocabulary from a pinned Civ VI gameplay DB.

Only content identifiers and native GameInfo indices are read. No saved-game,
player, map, timestamp, or entity-instance data enters this vocabulary.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path

from .encoding import ContentVocabulary, _token


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GAMEPLAY_DB = ROOT / "work/phase23-native-fixture01/DebugGameplay.sqlite"
DEFAULT_FIXTURE_MANIFEST = ROOT / "work/phase23-native-fixture01/manifest.json"
DEFAULT_GAMEPLAY_SHA256 = "4abe921833989c6c6f44f5a894de796c3b811ff7c0447d4858285c5fcf2bcd09"
DEFAULT_MANIFEST_SHA256 = "c435f291a0d1f91e840b24928155ebe5d2bb9a7978b499d086a7bbd450359539"
GAMEINFO_EVIDENCE_SHA256 = "61f762c40ab5a1f3789d4afdca711042d19083390ff800203bc3e091d8606a45"
GAMEINFO_EXE_SHA256 = "e7450823cc8e00468cff7b9d7b97c63140eae38ae1d774ba4efa437556c42d63"
GAMEINFO_EVIDENCE_PATH = ROOT / "evidence/phase41-gameinfo-maps.json"
MAX_DATABASE_BYTES = 1_073_741_824
MAX_TABLE_ROWS = 100_000
MAX_CONTENT_ROWS = 500_000
MAX_VOCABULARY_TOKENS = 100_000

# Primary key names are checked against SQLite PRAGMA metadata. GameInfo's
# zero-based Index comes from its source-pinned table iteration order; Hash is
# resolved via Types.Hash for that primary-key Type.
CONTENT_TABLES = {
    "Technologies": "TechnologyType",
    "Civics": "CivicType",
    "Policies": "PolicyType",
    "Units": "UnitType",
    "Buildings": "BuildingType",
    "Districts": "DistrictType",
    "Projects": "ProjectType",
    "Yields": "YieldType",
    "Resources": "ResourceType",
    "Improvements": "ImprovementType",
    "Routes": "RouteType",
    "Governments": "GovernmentType",
    "Beliefs": "BeliefType",
    "GreatPersonIndividuals": "GreatPersonIndividualType",
    "GreatPersonClasses": "GreatPersonClassType",
    "UnitPromotions": "UnitPromotionType",
    "UnitPromotionClasses": "PromotionClassType",
    "UnitOperations": "OperationType",
    "UnitCommands": "CommandType",
}
_TYPE_RE = re.compile(r"[A-Z][A-Z0-9_]{0,127}\Z")

_ACTION_ENUMS = {
    "action.category": ("unit", "building", "district", "project"),
    "action.mode": ("place", "resume", "append", "replace_tail"),
    "action.operation": ("MOVE_TO", "RANGE_ATTACK"),
    "action.input_route": ("enemy_flag", "dedicated_ranged", "world_plot"),
    "action.directive": ("KEEP", "RAZE", "LIBERATE_FOUNDER", "LIBERATE_PREVIOUS_OWNER"),
    "action.route": ("command", "operation"),
}


class ContentVocabularySourceError(ValueError):
    """The selected database cannot support a source-pinned public vocabulary."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quoted(identifier: str) -> str:
    if type(identifier) is not str or not identifier or "\x00" in identifier:
        raise ContentVocabularySourceError("invalid SQLite identifier")
    return '"' + identifier.replace('"', '""') + '"'


def _db_type(value, label: str) -> str:
    if type(value) is not str or _TYPE_RE.fullmatch(value) is None:
        raise ContentVocabularySourceError(f"invalid public content type in {label}")
    return value


def _db_hash(value, label: str) -> int:
    if type(value) is not int or not -(1 << 31) <= value <= 0xFFFFFFFF:
        raise ContentVocabularySourceError(f"invalid public content hash in {label}")
    return value & 0xFFFFFFFF


def _table_info(db, table: str):
    rows = db.execute("PRAGMA table_info(" + _quoted(table) + ")").fetchall()
    if not rows:
        raise ContentVocabularySourceError(f"required public table is missing: {table}")
    return rows


def _default_provenance(db_path: Path, digest: str) -> dict:
    manifest_path = DEFAULT_FIXTURE_MANIFEST
    if _sha256(manifest_path) != DEFAULT_MANIFEST_SHA256:
        raise ContentVocabularySourceError("pinned gameplay snapshot manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = manifest.get("databases", {}).get("DebugGameplay", {})
    copy_info = source.get("copy", {})
    if (manifest.get("schema") != "civ6rl.native-fixture.v1"
            or copy_info.get("sha256") != digest
            or source.get("source", {}).get("sha256") != digest
            or manifest.get("eligible_for_training") is not False
            or manifest.get("provenance", {}).get("ambiguous") is not True):
        raise ContentVocabularySourceError("pinned snapshot provenance no longer matches")
    return {
        "snapshot_manifest_sha256": DEFAULT_MANIFEST_SHA256,
        "source_path_label": "Firaxis Cache DebugGameplay.sqlite (byte-identical pinned copy)",
        "base_profile_qualification": "unqualified; snapshot provenance explicitly ambiguous",
        "training_fixture": False,
    }


def _extract_tokens(db) -> tuple[set[str], dict[str, int]]:
    """Extract bounded public tokens from an already-open SQLite connection."""
    tokens: set[str] = set()
    total_rows = 0
    table_counts = {}
    types_info = _table_info(db, "Types")
    types_columns = {row[1] for row in types_info}
    if not {"Type", "Hash"} <= types_columns:
        raise ContentVocabularySourceError("Types table lacks Type or Hash")
    types_pk = [row[1] for row in types_info if row[5]]
    if types_pk != ["Type"]:
        raise ContentVocabularySourceError("Types primary key differs from the pinned contract")
    type_hashes = {}
    for type_value, hash_value in db.execute('SELECT "Type","Hash" FROM "Types"'):
        type_name = _db_type(type_value, "Types.Type")
        if type_name in type_hashes:
            raise ContentVocabularySourceError("duplicate Types.Type")
        type_hashes[type_name] = _db_hash(hash_value, "Types.Hash")
    if not type_hashes or len(type_hashes) > MAX_CONTENT_ROWS:
        raise ContentVocabularySourceError("Types row count outside the declared bound")
    for type_name, content_hash in type_hashes.items():
        tokens.add(_token("action.item_type", type_name))
        tokens.add(_token("action.hash", content_hash))

    table_indices = {}
    for table, primary_key in CONTENT_TABLES.items():
        info = _table_info(db, table)
        columns = {row[1] for row in info}
        if primary_key not in columns:
            raise ContentVocabularySourceError(f"{table} lacks expected key column {primary_key}")
        primary_keys = [row[1] for row in info if row[5]]
        if primary_keys != [primary_key]:
            raise ContentVocabularySourceError(f"{table} primary-key metadata is not the expected field")
        query = "SELECT rowid," + _quoted(primary_key) + " FROM " + _quoted(table) + " ORDER BY rowid"
        rows = db.execute(query).fetchall()
        if len(rows) > MAX_TABLE_ROWS:
            raise ContentVocabularySourceError(f"{table} row count exceeds its bound")
        table_counts[table] = len(rows)
        total_rows += len(rows)
        if total_rows > MAX_CONTENT_ROWS:
            raise ContentVocabularySourceError("selected content row count exceeds its bound")
        seen_rowids, seen_names = set(), set()
        indexed = []
        for ordinal, (row_id, type_value) in enumerate(rows):
            type_name = _db_type(type_value, table + "." + primary_key)
            if type(row_id) is not int or row_id in seen_rowids or type_name in seen_names:
                raise ContentVocabularySourceError(f"duplicate or invalid row identity in {table}")
            seen_rowids.add(row_id)
            seen_names.add(type_name)
            if type_name not in type_hashes:
                raise ContentVocabularySourceError(f"{table} type is absent from Types.Hash: {type_name}")
            content_hash = type_hashes[type_name]
            indexed.append((type_name, ordinal, content_hash))
            tokens.add(_token("action.index", ordinal))
            if table == "UnitOperations" and type_name.startswith("UNITOPERATION_"):
                # UnitPanel work actions publish the public operation name as
                # their `operation` category and its GameInfo hash separately.
                tokens.add(_token("action.operation", type_name[len("UNITOPERATION_"):]))
            if table == "Units":
                tokens.add(_token("entity.type", ordinal))
            if table == "Policies":
                tokens.update((_token("policy.index", ordinal),
                               _token("policy.hash", content_hash),
                               _token("policy.type", type_name)))
        table_indices[table] = indexed

    for namespace, values in _ACTION_ENUMS.items():
        tokens.update(_token(namespace, value) for value in values)
    for table, field in (("Yields", "action.currency"), ("Yields", "action.yield_type"),
                         ("Routes", "action.route"), ("GreatPersonClasses", "action.class")):
        for type_name, _, _ in table_indices[table]:
            tokens.add(_token(field, type_name))
    for player_id in range(64):
        tokens.add(_token("action.target_player_id", player_id))
    if len(tokens) > MAX_VOCABULARY_TOKENS:
        raise ContentVocabularySourceError("public vocabulary token bound exceeded")
    table_counts["Types"] = len(type_hashes)
    return tokens, table_counts


def load_public_content_vocabulary(path: str | Path = DEFAULT_GAMEPLAY_DB, *,
                                   expected_sha256: str | None = None,
                                   source_label: str | None = None,
                                   include_action_identities: bool = False) -> ContentVocabulary:
    """Read one SHA-pinned gameplay SQLite database in read-only mode.

    For the checked-in Base-profile cache snapshot, provenance is verified
    against its pinned manifest. For a supplied database, the caller must pass
    its expected SHA-256; its Base/DLC/save qualification is never inferred.
    """
    db_path = Path(path).resolve(strict=True)
    if type(include_action_identities) is not bool:
        raise ContentVocabularySourceError('action identity extension must be explicitly boolean')
    if not db_path.is_file() or db_path.stat().st_size > MAX_DATABASE_BYTES:
        raise ContentVocabularySourceError("gameplay database missing or above the byte bound")
    pinned_digest = expected_sha256
    is_default = (db_path == DEFAULT_GAMEPLAY_DB.resolve() and expected_sha256 is None)
    if is_default:
        pinned_digest = DEFAULT_GAMEPLAY_SHA256
    if (type(pinned_digest) is not str or len(pinned_digest) != 64
            or any(char not in "0123456789abcdef" for char in pinned_digest)):
        raise ContentVocabularySourceError("a lowercase expected SHA-256 is required")
    before = _sha256(db_path)
    if before != pinned_digest:
        raise ContentVocabularySourceError("gameplay database SHA-256 differs from its pin")
    try:
        evidence_path = GAMEINFO_EVIDENCE_PATH.resolve(strict=True)
    except OSError as exc:
        raise ContentVocabularySourceError("pinned GameInfo mapping evidence is unavailable") from exc
    evidence_before = _sha256(evidence_path)
    if evidence_before != GAMEINFO_EVIDENCE_SHA256:
        raise ContentVocabularySourceError("pinned GameInfo mapping evidence changed")

    uri = db_path.as_uri() + "?mode=ro"
    try:
        db = sqlite3.connect(uri, uri=True)
        try:
            db.execute("PRAGMA query_only=ON")
            integrity = db.execute("PRAGMA quick_check(1)").fetchone()
            if integrity != ("ok",):
                raise ContentVocabularySourceError("gameplay database quick_check failed")
            tokens, table_counts = _extract_tokens(db)
        finally:
            db.close()
    except sqlite3.Error as exc:
        raise ContentVocabularySourceError("invalid or unreadable gameplay SQLite schema") from exc

    after = _sha256(db_path)
    if after != before:
        raise ContentVocabularySourceError("gameplay database changed during vocabulary extraction")
    if _sha256(evidence_path) != evidence_before:
        raise ContentVocabularySourceError("GameInfo mapping evidence changed during extraction")
    provenance = (_default_provenance(db_path, before) if is_default else {
        "source_path_label": source_label or "caller-supplied SHA-pinned gameplay SQLite",
        "base_profile_qualification": "unknown; no profile or save qualification inferred",
        "training_fixture": False,
    })
    manifest = {
        "schema": "civ6rl.learning.public-content-vocabulary.v1",
        "vocabulary_source": "public gameplay rules database",
        "database": {"sha256": before, "bytes": db_path.stat().st_size, **provenance},
        "gameinfo_index_contract": {
            "evidence": "evidence/phase41-gameinfo-maps.json",
            "evidence_sha256": GAMEINFO_EVIDENCE_SHA256,
            "evidence_sha256_verified_at_load": evidence_before,
            "exe_sha256": GAMEINFO_EXE_SHA256,
            "index": "zero-based ordinal from SELECT rowid,* ORDER BY rowid; not rowid-1",
            "hash": "Types.Hash looked up by the metadata-declared table primary-key Type",
        },
        "namespace_coverage": {
            "content_tables": dict(CONTENT_TABLES),
            "action_index": "all rows of the listed native action content tables",
            "entity_type": "Units GameInfo indices",
            "action_hash_item_type": "Types public content identities",
            "policy": ["policy.index", "policy.hash", "policy.type"],
            "other": sorted(_ACTION_ENUMS),
            "runtime_player_slots": "action.target_player_id fixed to native validated 0..63",
        },
        "limits": {
            "database_bytes": MAX_DATABASE_BYTES,
            "rows_per_table": MAX_TABLE_ROWS,
            "selected_content_rows": MAX_CONTENT_ROWS,
            "tokens": MAX_VOCABULARY_TOKENS,
            "rows_by_table": table_counts,
        },
        "excluded": ["save data", "player state", "map/fog state", "timestamps", "entity instance IDs"],
    }
    if include_action_identities:
        from .action_content import extract_action_content_catalog
        catalog = extract_action_content_catalog(db_path, expected_sha256=before)
        for table, namespace in (('Units', 'action.target_type'),
                                 ('GreatPersonIndividuals', 'action.individual')):
            tokens.update(_token(namespace, row['index']) for row in catalog['tables'][table])
        tokens.update(_token('action.class', row['class']) for row in catalog['tables']['Beliefs'])
        manifest['action_content'] = catalog
        manifest['namespace_coverage']['action_identities'] = {
            'action.target_type': 'Units GameInfo indices',
            'action.individual': 'GreatPersonIndividuals GameInfo indices',
            'action.class': 'GreatPersonClasses types and Beliefs.BeliefClassType'}
        if len(tokens) > MAX_VOCABULARY_TOKENS:
            raise ContentVocabularySourceError('extended vocabulary token bound exceeded')
    return ContentVocabulary(sorted(tokens), manifest=manifest)

