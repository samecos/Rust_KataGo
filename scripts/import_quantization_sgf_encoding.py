#!/usr/bin/env python3
"""Explicit, reversible FF4 Latin1 import for 19x19 Go SGFs.

This CPU-only tool interprets absent CA using the FF4 ISO-8859-1 default.
It does not detect character sets, recover historical names, repair SGF, replay
Go moves, or establish corpus readiness. Feed output/sgf to the native corpus
builder separately. Its longest-branch samples need not be actual-play mainlines.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
import sys
from typing import Any

SCHEMA = "rustgo-sgf-encoding-import-v1"
MODE = "ff4-default-or-explicit-iso8859-1-to-utf8-v1"
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_DEPTH = 4096
MAX_NODES = 100_000
WHITESPACE = " \t\r\n\v\f"
PROTECTED_KEYS = frozenset(("GM", "FF", "SZ", "RU", "KM", "HA", "PL", "AB", "AW", "AE", "B", "W", "AP", "US"))
# Deliberately limited to the text properties observed in the audited source.
NON_ASCII_TEXT_KEYS = frozenset(("PB", "PW", "BR", "WR", "GN", "C"))
SCALAR_KEYS = PROTECTED_KEYS - {"AB", "AW", "AE", "AP"}


class ImportFailure(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Conversion:
    data: bytes
    evidence: dict[str, Any]


@dataclass(frozen=True)
class _Property:
    name: str
    values: tuple[str, ...]
    raw_values: tuple[str, ...]
    value_spans: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class _Node:
    semicolon: int
    properties: tuple[_Property, ...]


@dataclass(frozen=True)
class _Tree:
    nodes: tuple[_Node, ...]
    children: tuple[_Tree, ...]


@dataclass
class _Frame:
    nodes: list[_Node]
    children: list[_Tree]
    children_started: bool = False


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode("ascii")


class _Parser:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.node_count = 0

    def _ws(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos] in WHITESPACE:
            self.pos += 1

    def _error(self, message: str) -> None:
        raise ImportFailure("sgf_syntax", f"{message} at character {self.pos}")

    def _value(self) -> tuple[str, tuple[int, int]]:
        self.pos += 1  # opening '['
        start = self.pos
        result: list[str] = []
        escaping = False
        while self.pos < len(self.text):
            c = self.text[self.pos]
            if not escaping and c == "]":
                end = self.pos
                self.pos += 1
                return "".join(result), (start, end)
            self.pos += 1
            if not escaping and c == "\\":
                escaping = True
                continue
            if c in "\r\n":
                while self.pos < len(self.text) and self.text[self.pos] in "\r\n":
                    self.pos += 1
                if not escaping:
                    result.append("\n")
            elif c in "\t\v\f":
                result.append(" ")
            else:
                result.append(c)
            escaping = False
        self._error("unterminated property value")

    def _node(self) -> _Node:
        semicolon = self.pos
        self.pos += 1
        self.node_count += 1
        if self.node_count > MAX_NODES:
            raise ImportFailure("resource_limit", "Node count exceeds supported limit")
        properties: list[_Property] = []
        self._ws()
        while self.pos < len(self.text) and "A" <= self.text[self.pos] <= "Z":
            start = self.pos
            while self.pos < len(self.text) and "A" <= self.text[self.pos] <= "Z":
                self.pos += 1
            name = self.text[start:self.pos]
            values, spans = [], []
            self._ws()
            while self.pos < len(self.text) and self.text[self.pos] == "[":
                value, span = self._value()
                values.append(value)
                spans.append(span)
                self._ws()
            if not values:
                self._error(f"property {name} has no values")
            properties.append(_Property(name, tuple(values),
                                        tuple(self.text[a:b] for a, b in spans), tuple(spans)))
        return _Node(semicolon, tuple(properties))

    def parse(self) -> _Tree:
        self._ws()
        if self.pos == len(self.text) or self.text[self.pos] != "(":
            self._error("expected opening tree parenthesis")
        self.pos += 1
        stack = [_Frame([], [])]
        while stack:
            self._ws()
            if self.pos == len(self.text):
                self._error("expected closing tree parenthesis")
            frame = stack[-1]
            c = self.text[self.pos]
            if c == ";":
                if frame.children_started:
                    self._error("sequence cannot continue after child variations")
                frame.nodes.append(self._node())
            elif c == "(":
                if not frame.nodes:
                    self._error("empty tree sequence")
                if len(stack) >= MAX_DEPTH:
                    raise ImportFailure("resource_limit", "Tree depth exceeds supported limit")
                frame.children_started = True
                self.pos += 1
                stack.append(_Frame([], []))
            elif c == ")":
                if not frame.nodes:
                    self._error("empty tree sequence")
                self.pos += 1
                tree = _Tree(tuple(frame.nodes), tuple(frame.children))
                stack.pop()
                if stack:
                    stack[-1].children.append(tree)
                else:
                    self._ws()
                    if self.pos != len(self.text):
                        self._error("trailing content or multiple games")
                    return tree
            else:
                self._error("expected node, child variation, or closing parenthesis")
        self._error("missing tree")


def _nodes(tree: _Tree):
    stack = [tree]
    while stack:
        current = stack.pop()
        yield from current.nodes
        stack.extend(reversed(current.children))


def _tree_value(tree: _Tree, *, omit_root_ca: bool = False, protected: bool = False,
                raw_spelling: bool = False) -> Any:
    # A flat ordered event stream also keeps JSON encoding and equality checks
    # independent of Python's recursion limit for deeply nested real SGFs.
    events = []
    stack: list[tuple[_Tree | None, bool]] = [(tree, True)]
    while stack:
        current, is_root = stack.pop()
        if current is None:
            events.append([")"])
            continue
        events.append(["("])
        for index, node in enumerate(current.nodes):
            events.append([";"])
            for p in node.properties:
                if omit_root_ca and is_root and index == 0 and p.name == "CA":
                    continue
                if protected and p.name not in PROTECTED_KEYS:
                    continue
                events.append(["property", p.name, list(p.raw_values if raw_spelling else p.values)])
        stack.append((None, False))
        stack.extend((child, False) for child in reversed(current.children))
    return events


def _validate(tree: _Tree) -> tuple[_Property | None, int]:
    root = tree.nodes[0]
    all_nodes = list(_nodes(tree))
    ff = [(node, p) for node in all_nodes for p in node.properties if p.name == "FF"]
    if len(ff) != 1 or ff[0][0] is not root or ff[0][1].values != ("4",):
        raise ImportFailure("format_version", "FF must occur once, as root FF[4]")
    ca = [(node, p) for node in all_nodes for p in node.properties if p.name == "CA"]
    if len(ca) > 1 or ca and (ca[0][0] is not root or len(ca[0][1].values) != 1):
        raise ImportFailure("encoding_declaration", "CA must be absent or a unique single-valued root property")
    if ca and ca[0][1].values[0].upper() != "ISO-8859-1":
        raise ImportFailure("unsupported_encoding", "Only absent CA or explicit CA[ISO-8859-1] is supported")
    duplicate_ap = 0
    for node in all_nodes:
        seen: set[str] = set()
        move_count = 0
        for p in node.properties:
            if p.name in seen:
                if p.name != "AP" or node is not root:
                    raise ImportFailure("duplicate_property", f"Duplicate {p.name}; only root AP repetition is supported")
                duplicate_ap += 1
            seen.add(p.name)
            if p.name in SCALAR_KEYS and len(p.values) != 1:
                raise ImportFailure("semantic_arity", f"{p.name} must have one value")
            if p.name in ("B", "W"):
                move_count += len(p.values)
            for value in p.values:
                if p.name in PROTECTED_KEYS and not value.isascii():
                    raise ImportFailure("non_ascii_semantics", f"Non-ASCII protected property {p.name}")
                if not value.isascii() and p.name not in NON_ASCII_TEXT_KEYS:
                    raise ImportFailure("unsupported_non_ascii_property", f"Non-ASCII {p.name} is outside this importer scope")
        if move_count > 1:
            raise ImportFailure("semantic_arity", "A node cannot contain multiple moves")
    root_props = {p.name: p.values for p in root.properties}
    if root_props.get("GM") != ("1",) or root_props.get("SZ") != ("19",):
        raise ImportFailure("unsupported_board_game", "This importer requires explicit root GM[1]SZ[19]")
    return ca[0][1] if ca else None, duplicate_ap


def recover_source_bytes(data: bytes, evidence: dict[str, Any]) -> bytes:
    """Verify a frozen derivative and reverse only its recorded CA operation."""
    try:
        if _sha(data) != evidence["derived_sha256"]:
            raise ValueError("derivative hash differs")
        op = evidence["operation"]
        text = data.decode("utf-8", errors="strict")
        start, end = op["text_start"], op["text_end"]
        if type(start) is not int or type(end) is not int or not 0 <= start <= end <= len(text):
            raise ValueError("invalid reversal offsets")
        if text[start:end] != op["replacement_text"]:
            raise ValueError("encoding operation differs")
        original = bytes.fromhex(op["source_bytes_hex"]).decode("iso8859-1", errors="strict")
        recovered = (text[:start] + original + text[end:]).encode("iso8859-1", errors="strict")
        if _sha(recovered) != evidence["source_sha256"]:
            raise ValueError("recovered source hash differs")
        return recovered
    except (KeyError, TypeError, ValueError, UnicodeError) as exc:
        raise ImportFailure("reversal_failed", str(exc)) from exc


def convert_sgf_bytes(raw: bytes) -> Conversion:
    if len(raw) > MAX_FILE_BYTES:
        raise ImportFailure("size_limit", f"SGF exceeds {MAX_FILE_BYTES} bytes")
    # Latin1 is required by this explicit mode, never guessed from readable text.
    text = raw.decode("iso8859-1", errors="strict")
    if text.encode("iso8859-1", errors="strict") != raw:
        raise ImportFailure("reversal_failed", "initial byte mapping did not roundtrip")
    before = _Parser(text).parse()
    ca, duplicates = _validate(before)
    if ca is None:
        start = end = before.nodes[0].semicolon + 1
        replacement, kind = "CA[UTF-8]", "insert-ca"
    else:
        start, end = ca.value_spans[0]
        replacement, kind = "UTF-8", "replace-ca-value"
    edited = text[:start] + replacement + text[end:]
    data = edited.encode("utf-8", errors="strict")
    after = _Parser(data.decode("utf-8", errors="strict")).parse()
    original_tree = _tree_value(before, omit_root_ca=True)
    converted_tree = _tree_value(after, omit_root_ca=True)
    if original_tree != converted_tree:
        raise ImportFailure("semantic_mismatch", "Full tree/property values changed beyond root CA")
    original_raw_tree = _tree_value(before, omit_root_ca=True, raw_spelling=True)
    if original_raw_tree != _tree_value(after, omit_root_ca=True, raw_spelling=True):
        raise ImportFailure("semantic_mismatch", "Original escape spelling or tree structure changed")
    protected = _tree_value(before, protected=True)
    if protected != _tree_value(after, protected=True):
        raise ImportFailure("semantic_mismatch", "Protected NN fields changed")
    evidence = {
        "mode": MODE, "source_sha256": _sha(raw), "derived_sha256": _sha(data),
        "source_encoding_basis": "FF4_CA_default" if ca is None else "explicit_ISO-8859-1",
        "operation": {"kind": kind, "source_byte_start": start, "source_byte_end": end,
                      "source_bytes_hex": raw[start:end].hex(), "text_start": start,
                      "text_end": start + len(replacement), "replacement_text": replacement},
        "whole_tree_equivalent_except_encoding_declaration": True,
        "semantic_evidence_encoding": "ordered-flat-sgf-events-json-v1",
        "whole_tree_without_root_ca_sha256": _sha(_json_bytes(original_tree)),
        "raw_escaped_tree_without_root_ca_sha256": _sha(_json_bytes(original_raw_tree)),
        "original_escape_spelling_equal": True,
        "protected_fields_ascii_and_equal": True,
        "protected_fields_sha256": _sha(_json_bytes(protected)),
        "tree_count": 1, "node_count_all_branches": sum(1 for _ in _nodes(before)),
        "duplicate_ap_occurrences": duplicates, "original_bytes_recoverable": True,
    }
    if recover_source_bytes(data, evidence) != raw:
        raise ImportFailure("reversal_failed", "Final exact byte check failed")
    return Conversion(data, evidence)


def validate_relative_path(name: str) -> None:
    if not name or any(c in name for c in '\\:<>"|?*') or any(ord(c) < 32 for c in name):
        raise ImportFailure("unsafe_path", f"Unsupported relative path: {name!r}")
    if PurePosixPath(name).is_absolute() or PureWindowsPath(name).is_absolute():
        raise ImportFailure("unsafe_path", "Absolute member path")
    for part in name.split("/"):
        if part in ("", ".", "..") or part.endswith((" ", ".")):
            raise ImportFailure("unsafe_path", "Traversal or ambiguous path component")
        if re.fullmatch(r"(?i:CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³])", part.split(".")[0].rstrip(" .")):
            raise ImportFailure("unsafe_path", "Windows reserved path component")


def _is_link(st: os.stat_result) -> bool:
    return stat.S_ISLNK(st.st_mode) or bool(getattr(st, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _check_path_chain(path: Path) -> None:
    for p in [path, *path.parents]:
        try:
            st = p.lstat()
        except FileNotFoundError:
            continue
        if _is_link(st):
            raise ImportFailure("linked_path", f"Symlink/reparse/junction path rejected: {p}")


def _identity(st: os.stat_result) -> tuple[int, int, int, int]:
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns


def _inventory(root: Path) -> tuple[list[tuple[str, os.stat_result]], list[str]]:
    files, ignored, seen = [], [], set()

    def walk(directory: Path) -> None:
        for entry in sorted(os.scandir(directory), key=lambda e: (e.name.casefold(), e.name)):
            p = Path(entry.path)
            relative = p.relative_to(root).as_posix()
            validate_relative_path(relative)
            if relative.casefold() in seen:
                raise ImportFailure("unsafe_path", "Case-colliding input paths")
            seen.add(relative.casefold())
            # Windows DirEntry.stat() may expose zero st_dev/st_ino; lstat()
            # provides the file identity later compared against fstat().
            st = p.lstat()
            if _is_link(st):
                raise ImportFailure("linked_path", f"Linked input rejected: {relative}")
            if stat.S_ISDIR(st.st_mode):
                walk(p)
            elif stat.S_ISREG(st.st_mode):
                if st.st_nlink > 1:
                    raise ImportFailure("linked_path", f"Hardlinked input rejected: {relative}")
                if p.suffix.lower() == ".sgf":
                    if st.st_size > MAX_FILE_BYTES:
                        raise ImportFailure("size_limit", f"Oversized input: {relative}")
                    files.append((relative, st))
                else:
                    ignored.append(relative)
            else:
                raise ImportFailure("unsafe_path", f"Nonregular input rejected: {relative}")
    walk(root)
    return files, ignored


def _read_source(path: Path, expected: os.stat_result) -> bytes:
    _check_path_chain(path)
    before = path.lstat()
    if _is_link(before) or not stat.S_ISREG(before.st_mode) or _identity(before) != _identity(expected):
        raise ImportFailure("source_changed", f"Input changed after inventory: {path}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(path, flags), "rb") as f:
        opened = os.fstat(f.fileno())
        if _identity(opened) != _identity(before) or _is_link(opened):
            raise ImportFailure("source_changed", f"Input changed during open: {path}")
        data = f.read(MAX_FILE_BYTES + 1)
        after = os.fstat(f.fileno())
    if len(data) > MAX_FILE_BYTES or _identity(after) != _identity(before) or _identity(path.lstat()) != _identity(before):
        raise ImportFailure("source_changed", f"Input changed while reading: {path}")
    return data


def _write_new(path: Path, data: bytes) -> None:
    _check_path_chain(path.parent)
    with path.open("xb") as f:
        f.write(data)


def import_directory(input_dir: Path | str, output_dir: Path | str) -> dict[str, Any]:
    source_arg, output_arg = Path(input_dir).absolute(), Path(output_dir).absolute()
    _check_path_chain(source_arg)
    _check_path_chain(output_arg)
    source, output = source_arg.resolve(), output_arg.resolve()
    if not source.is_dir():
        raise ImportFailure("input_directory", "Input must be an existing regular directory")
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ImportFailure("overlapping_paths", "Input and output directory trees must not overlap")
    if output.exists():
        raise ImportFailure("output_exists", "Output must be a new directory")
    inventory, ignored = _inventory(source)
    if not inventory:
        raise ImportFailure("empty_input", "No regular .sgf files found")
    output.parent.mkdir(parents=True, exist_ok=True)
    _check_path_chain(output.parent)
    output.mkdir(exist_ok=False)
    records = []
    for relative, original_stat in inventory:
        raw = _read_source(source / relative, original_stat)
        record: dict[str, Any] = {"source": {"relative_path": relative, "bytes": len(raw), "sha256": _sha(raw)}}
        try:
            result = convert_sgf_bytes(raw)
        except ImportFailure as exc:
            record.update(status="rejected", rejection={"code": exc.code, "detail": str(exc)})
        else:
            destination = output / "sgf" / relative
            if not destination.resolve().is_relative_to(output):
                raise ImportFailure("unsafe_path", "Destination escaped output directory")
            destination.parent.mkdir(parents=True, exist_ok=True)
            _write_new(destination, result.data)
            record.update(status="accepted", derived={"relative_path": destination.relative_to(output).as_posix(),
                          "bytes": len(result.data), "sha256": _sha(result.data)}, evidence=result.evidence)
        records.append(record)
    # Recheck every original, including rejected files, before final evidence.
    for (relative, original_stat), record in zip(inventory, records):
        if _sha(_read_source(source / relative, original_stat)) != record["source"]["sha256"]:
            raise ImportFailure("source_changed", f"Input content changed: {relative}")
    accepted = sum(r["status"] == "accepted" for r in records)
    manifest = {
        "schema": SCHEMA, "mode": MODE, "status": "ENCODING_IMPORT_ONLY",
        "input_directory": str(source), "output_directory": str(output),
        "counts": {"sources": len(records), "accepted": accepted, "rejected": len(records)-accepted,
                   "ignored_regular_files": len(ignored)},
        "records": records, "ignored_regular_files": ignored,
        "protected_keys": sorted(PROTECTED_KEYS), "non_ascii_text_keys": sorted(NON_ASCII_TEXT_KEYS),
        "resource_limits": {"max_file_bytes": MAX_FILE_BYTES, "max_tree_levels": MAX_DEPTH, "max_nodes": MAX_NODES},
        "source_identity_scope": "Original member relative paths and exact byte SHA256; bind archive provenance separately when applicable.",
        "equivalence_scope": "Complete ordered trees, all branches, all property values and original escape spelling, excluding only the recorded root CA operation.",
        "compatibility_limit": "Repeated root AP is a native Fox-compatible extension, not strict FF4 conformance certification. All other repeated properties are rejected.",
        "text_encoding_limit": "FF4 default/explicit Latin1 interpretation only; no claim to identify or recover historical names/comments.",
        "sample_scope": "Encoding evidence only. Native builder replay and longest-branch selection are separate; a selected longest branch need not be the actual-play mainline.",
        "native_replay_performed": False, "corpus_readiness": "NOT_EVALUATED", "model_outputs_accessed": False,
        "original_files_unchanged_verified": True,
        "tool_sha256": _sha(Path(__file__).read_bytes()), "python_version": sys.version,
        "specification": "https://www.red-bean.com/sgf/properties.html#CA",
    }
    _write_new(output / "manifest.json", (json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Original SGF directory; never modified")
    parser.add_argument("--output", type=Path, required=True, help="New, separate directory for manifest.json and sgf/")
    args = parser.parse_args(argv)
    try:
        manifest = import_directory(args.input, args.output)
    except (ImportFailure, OSError) as exc:
        print(f"SGF encoding import failed [{getattr(exc, 'code', 'filesystem')}]: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": manifest["status"], "counts": manifest["counts"],
                      "manifest": str(args.output.resolve() / "manifest.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
