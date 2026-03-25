# mcp-codebase-index - Structural codebase indexer with MCP server
# Copyright (C) 2026 Michael Doyle
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# Commercial licensing available. See COMMERCIAL-LICENSE.md for details.

"""MCP server for the structural codebase indexer.

Exposes project-wide structural query functions as MCP tools,
enabling Claude Code to navigate codebases efficiently without
reading entire files into context.

Usage:
    PROJECT_ROOT=/path/to/project python -m mcp_codebase_index.server
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import sys
import pickle
import time
import traceback

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import mcp.types as types

from mcp_codebase_index.git_tracker import is_git_repo, get_head_commit, get_changed_files
from mcp_codebase_index.models import ProjectIndex
from mcp_codebase_index.project_indexer import ProjectIndexer
from mcp_codebase_index.query_api import create_project_query_functions

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

server = Server("mcp-codebase-index")

_project_root: str = ""
_indexer: ProjectIndexer | None = None
_query_fns: dict | None = None
_is_git: bool = False

# Persistent cache
_CACHE_FILENAME = ".codebase-index-cache.pkl"
_CACHE_VERSION = 1  # Bump when ProjectIndex schema changes

# Session usage stats
_session_start: float = time.time()
_tool_call_counts: dict[str, int] = {}
_total_chars_returned: int = 0

# Persistent stats
_STATS_DIR = os.path.expanduser("~/.local/share/mcp-codebase-index")
_stats_file: str = ""  # Set after PROJECT_ROOT is known


def _get_stats_file(project_root: str) -> str:
    """Return path to the stats JSON file for this project."""
    slug = hashlib.md5(project_root.encode()).hexdigest()[:8]
    name = os.path.basename(project_root.rstrip("/"))
    return os.path.join(_STATS_DIR, f"{name}-{slug}.json")


def _load_cumulative_stats() -> dict:
    """Load cumulative stats from disk, or return empty structure."""
    if not _stats_file or not os.path.exists(_stats_file):
        return {"total_calls": 0, "total_chars_returned": 0, "total_naive_chars": 0, "sessions": 0, "tool_counts": {}}
    try:
        with open(_stats_file) as f:
            return json.load(f)
    except Exception:
        return {"total_calls": 0, "total_chars_returned": 0, "total_naive_chars": 0, "sessions": 0, "tool_counts": {}}


def _flush_stats(naive_chars: int) -> None:
    """Append current session stats to the persistent JSON file."""
    if not _stats_file:
        return
    try:
        os.makedirs(_STATS_DIR, exist_ok=True)
        cum = _load_cumulative_stats()
        session_calls = sum(_tool_call_counts.values()) - _tool_call_counts.get("get_usage_stats", 0)
        cum["sessions"] = cum.get("sessions", 0) + 1
        cum["total_calls"] = cum.get("total_calls", 0) + session_calls
        cum["total_chars_returned"] = cum.get("total_chars_returned", 0) + _total_chars_returned
        cum["total_naive_chars"] = cum.get("total_naive_chars", 0) + naive_chars
        cum["project"] = _project_root
        cum["last_session"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for tool, count in _tool_call_counts.items():
            if tool == "get_usage_stats":
                continue
            cum["tool_counts"][tool] = cum["tool_counts"].get(tool, 0) + count
        with open(_stats_file, "w") as f:
            json.dump(cum, f, indent=2)
    except Exception as e:
        print(f"[mcp-codebase-index] Failed to flush stats: {e}", file=sys.stderr)

# Realistic estimate of what % of codebase you'd need to read without the indexer
_TOOL_COST_MULTIPLIERS: dict[str, float] = {
    "get_project_summary": 0.10,
    "list_files": 0.01,
    "get_structure_summary": 0.05,
    "get_functions": 0.05,
    "get_classes": 0.05,
    "get_imports": 0.03,
    "get_function_source": 0.02,
    "get_class_source": 0.03,
    "find_symbol": 0.05,
    "get_dependencies": 0.10,
    "get_dependents": 0.15,
    "get_change_impact": 0.30,
    "get_call_chain": 0.20,
    "get_file_dependencies": 0.02,
    "get_file_dependents": 0.10,
    "search_codebase": 0.15,
    "reindex": 0.0,
    "set_project_root": 0.0,
}


def _format_result(value: object) -> str:
    """Format a query result as readable text."""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, indent=2, default=str)
    return str(value)


def _format_usage_stats(include_cumulative: bool = False) -> str:
    """Format session usage statistics, optionally with cumulative history."""
    elapsed = time.time() - _session_start
    total_calls = sum(_tool_call_counts.values())
    # Don't count get_usage_stats itself in the query total
    query_calls = total_calls - _tool_call_counts.get("get_usage_stats", 0)

    # Calculate total source size from the index
    source_chars = 0
    if _indexer and _indexer._project_index:
        source_chars = sum(m.total_chars for m in _indexer._project_index.files.values())

    lines = [
        f"Session duration: {_format_duration(elapsed)}",
        f"Total queries: {query_calls}",
    ]

    if _tool_call_counts:
        lines.append("")
        lines.append("Queries by tool:")
        for tool_name, count in sorted(_tool_call_counts.items(), key=lambda x: -x[1]):
            if tool_name == "get_usage_stats":
                continue
            lines.append(f"  {tool_name}: {count}")

    lines.append("")
    lines.append(f"Total chars returned: {_total_chars_returned:,}")

    if source_chars > 0:
        lines.append(f"Total source in index: {source_chars:,} chars")
        if query_calls > 0 and source_chars > _total_chars_returned:
            # Per-tool estimate of what you'd read without the indexer
            naive_chars = 0
            for tool_name, count in _tool_call_counts.items():
                if tool_name == "get_usage_stats":
                    continue
                multiplier = _TOOL_COST_MULTIPLIERS.get(tool_name, 0.10)
                naive_chars += int(source_chars * multiplier * count)
            reduction = (1 - _total_chars_returned / naive_chars) * 100 if naive_chars > 0 else 0
            lines.append(
                f"Estimated without indexer: {naive_chars:,} chars "
                f"({naive_chars // 4:,} tokens) over {query_calls} queries"
            )
            lines.append(
                f"Estimated with indexer: {_total_chars_returned:,} chars "
                f"({_total_chars_returned // 4:,} tokens)"
            )
            lines.append(f"Estimated token savings: {reduction:.1f}%")

    if include_cumulative:
        cum = _load_cumulative_stats()
        cum_calls = cum.get("total_calls", 0)
        if cum_calls > 0:
            lines.append("")
            lines.append("─── Cumulative (all sessions) ───")
            lines.append(f"Sessions: {cum.get('sessions', 0)}")
            lines.append(f"Total queries: {cum_calls:,}")
            cum_chars = cum.get("total_chars_returned", 0)
            cum_naive = cum.get("total_naive_chars", 0)
            lines.append(f"Chars returned: {cum_chars:,} ({cum_chars // 4:,} tokens)")
            if cum_naive > 0:
                cum_reduction = (1 - cum_chars / cum_naive) * 100 if cum_naive > cum_chars else 0
                lines.append(f"Naive estimate: {cum_naive:,} ({cum_naive // 4:,} tokens)")
                lines.append(f"Token savings: {cum_reduction:.1f}%")
            if cum.get("tool_counts"):
                lines.append("Top tools:")
                for t, c in sorted(cum["tool_counts"].items(), key=lambda x: -x[1])[:5]:
                    lines.append(f"  {t}: {c:,}")
            if cum.get("last_session"):
                lines.append(f"Last session: {cum['last_session']}")

    return "\n".join(lines)


def _format_duration(seconds: float) -> str:
    """Format seconds into a human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins}m"


def _cache_path(project_root: str) -> str:
    """Return the path to the pickle cache file for this project."""
    return os.path.join(project_root, _CACHE_FILENAME)


def _save_cache(index: "ProjectIndex") -> None:
    """Persist the project index to a pickle cache file."""
    try:
        root = index.root_path
        path = _cache_path(root)
        payload = {"version": _CACHE_VERSION, "index": index}
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[mcp-codebase-index] Cache saved → {path}", file=sys.stderr)
    except Exception as exc:
        print(f"[mcp-codebase-index] Cache save failed: {exc}", file=sys.stderr)


def _load_cache(project_root: str) -> "ProjectIndex | None":
    """Load a cached project index if it exists and is compatible."""
    path = _cache_path(project_root)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict) or payload.get("version") != _CACHE_VERSION:
            print("[mcp-codebase-index] Cache version mismatch, ignoring", file=sys.stderr)
            return None
        index = payload["index"]
        from mcp_codebase_index.models import ProjectIndex as PI
        if not isinstance(index, PI):
            return None
        return index
    except Exception as exc:
        print(f"[mcp-codebase-index] Cache load failed: {exc}", file=sys.stderr)
        return None


def _ensure_index() -> None:
    """Build the project index on first use (lazy initialization).

    Tries to load from a pickle cache first. If the cache is valid and
    the git ref matches (or the changeset is small enough for incremental
    update), skips a full rebuild.

    This is called on the first tool call rather than at startup so that
    the MCP server can complete its initialization handshake immediately.
    Without this, large projects would cause Claude Code to timeout waiting
    for the server to become ready.
    """
    global _project_root, _indexer, _query_fns, _is_git, _stats_file

    if _indexer is not None:
        return

    env_root = os.environ.get("PROJECT_ROOT", "").strip()
    if not env_root and not _project_root:
        # Generic instance with no project assigned — stay idle until set_project_root is called
        return

    if not _project_root:
        _project_root = env_root

    _is_git = is_git_repo(_project_root)
    if not _stats_file:
        _stats_file = _get_stats_file(_project_root)

    cached_index = _load_cache(_project_root)
    if cached_index is not None and _is_git and cached_index.last_indexed_git_ref:
        current_head = get_head_commit(_project_root)
        if current_head == cached_index.last_indexed_git_ref:
            # Exact match — use cache directly
            print("[mcp-codebase-index] Cache hit (git ref matches)", file=sys.stderr)
            _indexer = ProjectIndexer(_project_root)
            _indexer._project_index = cached_index
            _query_fns = create_project_query_functions(cached_index)
            return

        # Check if changeset is small enough for incremental update on cache
        changeset = get_changed_files(_project_root, cached_index.last_indexed_git_ref)
        total_changes = len(changeset.modified) + len(changeset.added) + len(changeset.deleted)
        if not changeset.is_empty and total_changes <= 20:
            print(
                f"[mcp-codebase-index] Cache hit with {total_changes} changed files, "
                f"applying incremental update",
                file=sys.stderr,
            )
            _indexer = ProjectIndexer(_project_root)
            _indexer._project_index = cached_index
            _query_fns = create_project_query_functions(cached_index)
            # _maybe_incremental_update will handle the rest on first tool call
            return

        print(
            f"[mcp-codebase-index] Cache stale ({total_changes} changes), full rebuild",
            file=sys.stderr,
        )

    _build_index()


def _build_index() -> None:
    """Build (or rebuild) the project index and query functions."""
    global _project_root, _indexer, _query_fns, _is_git, _stats_file

    if not _project_root:
        _project_root = os.environ.get("PROJECT_ROOT", os.getcwd())

    if not _stats_file:
        _stats_file = _get_stats_file(_project_root)
    print(f"[mcp-codebase-index] Indexing project: {_project_root}", file=sys.stderr)

    # Allow per-project pattern overrides via env vars (colon-separated globs)
    # EXCLUDE_EXTRA: additional patterns appended to the defaults
    # EXCLUDE_PATTERNS: full override of exclude patterns
    # INCLUDE_PATTERNS: full override of include patterns
    extra_excludes_raw = os.environ.get("EXCLUDE_EXTRA", "")
    exclude_override_raw = os.environ.get("EXCLUDE_PATTERNS", "")
    include_override_raw = os.environ.get("INCLUDE_PATTERNS", "")

    exclude_patterns = None
    include_patterns = None

    if exclude_override_raw:
        exclude_patterns = [p.strip() for p in exclude_override_raw.split(":") if p.strip()]
    elif extra_excludes_raw:
        # Start from defaults, append extras
        tmp = ProjectIndexer(_project_root)
        exclude_patterns = tmp.exclude_patterns + [p.strip() for p in extra_excludes_raw.split(":") if p.strip()]

    if include_override_raw:
        include_patterns = [p.strip() for p in include_override_raw.split(":") if p.strip()]

    _indexer = ProjectIndexer(_project_root, include_patterns=include_patterns, exclude_patterns=exclude_patterns)
    index = _indexer.index()
    _query_fns = create_project_query_functions(index)

    if not _is_git:
        _is_git = is_git_repo(_project_root)
    if _is_git:
        index.last_indexed_git_ref = get_head_commit(_project_root)
        _save_cache(index)

    print(
        f"[mcp-codebase-index] Indexed {index.total_files} files, "
        f"{index.total_lines} lines, "
        f"{index.total_functions} functions, "
        f"{index.total_classes} classes "
        f"in {index.index_build_time_seconds:.2f}s",
        file=sys.stderr,
    )


def _matches_include_patterns(rel_path: str, patterns: list[str]) -> bool:
    """Check if a relative path matches any of the include glob patterns."""
    normalized = rel_path.replace(os.sep, "/")
    for pattern in patterns:
        if fnmatch.fnmatch(normalized, pattern):
            return True
    return False


def _maybe_incremental_update() -> None:
    """Check git for changes and incrementally update the index if needed."""
    if not _is_git or _indexer is None or _indexer._project_index is None:
        return

    idx = _indexer._project_index
    changeset = get_changed_files(_project_root, idx.last_indexed_git_ref)
    if changeset.is_empty:
        return

    total_changes = len(changeset.modified) + len(changeset.added) + len(changeset.deleted)

    # Large changeset threshold: full rebuild for branch switches etc.
    if total_changes > 20 and total_changes > idx.total_files * 0.5:
        print(
            f"[mcp-codebase-index] Large changeset ({total_changes} files), "
            f"doing full rebuild",
            file=sys.stderr,
        )
        _build_index()
        return

    # Process deletions
    for path in changeset.deleted:
        if path in idx.files:
            _indexer.remove_file(path)

    # Process modifications and additions
    for path in changeset.modified + changeset.added:
        if _indexer._is_excluded(path):
            continue
        if not _matches_include_patterns(path, _indexer.include_patterns):
            continue
        abs_path = os.path.join(_project_root, path)
        if not os.path.isfile(abs_path):
            continue
        _indexer.reindex_file(path, skip_graph_rebuild=True)

    # Rebuild cross-file graphs once
    _indexer.rebuild_graphs()

    # Update the git ref
    idx.last_indexed_git_ref = get_head_commit(_project_root)

    n_mod = len(changeset.modified)
    n_add = len(changeset.added)
    n_del = len(changeset.deleted)
    print(
        f"[mcp-codebase-index] Incremental update: "
        f"{n_mod} modified, {n_add} added, {n_del} deleted",
        file=sys.stderr,
    )

    _save_cache(idx)


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS = [
    Tool(
        name="get_project_summary",
        description="High-level overview of the project: file count, packages, top classes/functions.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="list_files",
        description="List indexed files. Optional glob pattern to filter (e.g. '*.py', 'src/**/*.ts').",
        inputSchema={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to filter files (uses fnmatch).",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (0 = unlimited, default 0).",
                },
            },
        },
    ),
    Tool(
        name="get_structure_summary",
        description="Structure summary for a file (functions, classes, imports, line counts) or the whole project if no file specified.",
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Relative path to a file in the project. Omit for project-level summary.",
                },
            },
        },
    ),
    Tool(
        name="get_function_source",
        description="Get the full source code of a function or method by name. Uses the symbol table to locate the file automatically.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Function or method name (e.g. 'my_func' or 'MyClass.my_method').",
                },
                "file_path": {
                    "type": "string",
                    "description": "Optional file path to narrow the search.",
                },
                "max_lines": {
                    "type": "integer",
                    "description": "Maximum number of source lines to return (0 = unlimited, default 0).",
                },
            },
            "required": ["name"],
        },
    ),
    Tool(
        name="get_class_source",
        description="Get the full source code of a class by name. Uses the symbol table to locate the file automatically.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Class name.",
                },
                "file_path": {
                    "type": "string",
                    "description": "Optional file path to narrow the search.",
                },
                "max_lines": {
                    "type": "integer",
                    "description": "Maximum number of source lines to return (0 = unlimited, default 0).",
                },
            },
            "required": ["name"],
        },
    ),
    Tool(
        name="get_functions",
        description="List all functions (with name, lines, params, file). Filter to a specific file or get all project functions.",
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Relative path to filter to a single file. Omit for all project functions.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (0 = unlimited, default 0).",
                },
            },
        },
    ),
    Tool(
        name="get_classes",
        description="List all classes (with name, lines, methods, bases, file). Filter to a specific file or get all project classes.",
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Relative path to filter to a single file. Omit for all project classes.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (0 = unlimited, default 0).",
                },
            },
        },
    ),
    Tool(
        name="get_imports",
        description="List all imports (with module, names, line). Filter to a specific file or get all project imports.",
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Relative path to filter to a single file. Omit for all project imports.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (0 = unlimited, default 0).",
                },
            },
        },
    ),
    Tool(
        name="find_symbol",
        description="Find where a symbol (function, method, class) is defined. Returns file path, line range, type, signature, and a source preview (~20 lines).",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Symbol name to find (e.g. 'ProjectIndexer', 'annotate', 'MyClass.run').",
                },
            },
            "required": ["name"],
        },
    ),
    Tool(
        name="get_dependencies",
        description="What does this symbol call/use? Returns list of symbols referenced by the named function or class.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Symbol name to query.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (0 = unlimited, default 0).",
                },
            },
            "required": ["name"],
        },
    ),
    Tool(
        name="get_dependents",
        description="What calls/uses this symbol? Returns list of symbols that reference the named function or class.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Symbol name to query.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (0 = unlimited, default 0).",
                },
            },
            "required": ["name"],
        },
    ),
    Tool(
        name="get_change_impact",
        description="Analyze the impact of changing a symbol. Returns direct dependents and transitive (cascading) dependents.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Symbol name to analyze.",
                },
                "max_direct": {
                    "type": "integer",
                    "description": "Maximum number of direct dependents to return (0 = unlimited, default 0).",
                },
                "max_transitive": {
                    "type": "integer",
                    "description": "Maximum number of transitive dependents to return (0 = unlimited, default 0).",
                },
            },
            "required": ["name"],
        },
    ),
    Tool(
        name="get_call_chain",
        description="Find the shortest dependency path between two symbols (BFS through the dependency graph).",
        inputSchema={
            "type": "object",
            "properties": {
                "from_name": {
                    "type": "string",
                    "description": "Starting symbol name.",
                },
                "to_name": {
                    "type": "string",
                    "description": "Target symbol name.",
                },
            },
            "required": ["from_name", "to_name"],
        },
    ),
    Tool(
        name="get_file_dependencies",
        description="List files that this file imports from (file-level import graph).",
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Relative path to the file.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (0 = unlimited, default 0).",
                },
            },
            "required": ["file_path"],
        },
    ),
    Tool(
        name="get_file_dependents",
        description="List files that import from this file (reverse import graph).",
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Relative path to the file.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (0 = unlimited, default 0).",
                },
            },
            "required": ["file_path"],
        },
    ),
    Tool(
        name="search_codebase",
        description="Regex search across all indexed files. Returns up to 100 matches with file, line number, and content.",
        inputSchema={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regular expression pattern to search for.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 100, 0 = unlimited).",
                },
            },
            "required": ["pattern"],
        },
    ),
    Tool(
        name="reindex",
        description="Re-index the entire project. Use after making significant file changes to refresh the structural index.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="set_project_root",
        description=(
            "Switch the indexer to a different project directory. "
            "Use this on a generic (no PROJECT_ROOT env var) instance to point it at any codebase. "
            "Triggers a full reindex of the new root. "
            "After calling this, all other tools operate on the new project."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the project root directory.",
                },
            },
            "required": ["path"],
        },
    ),
    Tool(
        name="get_usage_stats",
        description="Session efficiency stats: tool calls, characters returned vs total source, estimated token savings.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
]


# ---------------------------------------------------------------------------
# MCP handlers
# ---------------------------------------------------------------------------


@server.list_tools()
async def list_tools() -> list[Tool]:
    return TOOLS


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    global _query_fns, _total_chars_returned

    # Track tool call counts (including reindex/stats themselves)
    _tool_call_counts[name] = _tool_call_counts.get(name, 0) + 1

    try:
        # Lazy initialization: build the index on first tool call so the
        # MCP handshake completes without waiting for indexing.
        _ensure_index()

        # Handle reindex separately since it rebuilds state
        if name == "reindex":
            _build_index()
            return [TextContent(type="text", text="Project re-indexed successfully.")]

        # Switch to a different project root
        if name == "set_project_root":
            new_root = os.path.abspath(arguments["path"])
            if not os.path.isdir(new_root):
                return [TextContent(type="text", text=f"Error: '{new_root}' is not a directory.")]
            global _project_root, _indexer, _stats_file
            _project_root = new_root
            _indexer = None  # Force full rebuild
            _stats_file = _get_stats_file(new_root)
            _build_index()
            return [TextContent(type="text", text=f"Switched to '{new_root}' and re-indexed successfully.")]

        # Handle usage stats
        if name == "get_usage_stats":
            return [TextContent(type="text", text=_format_usage_stats(include_cumulative=True))]

        _maybe_incremental_update()

        if _query_fns is None:
            if not _project_root:
                return [TextContent(type="text", text="Error: no project assigned. Call set_project_root('/path/to/project') first.")]
            return [TextContent(type="text", text="Error: index not built yet. Call reindex first.")]

        # Dispatch to the appropriate query function
        if name == "get_project_summary":
            result = _query_fns["get_project_summary"]()

        elif name == "list_files":
            pattern = arguments.get("pattern")
            max_results = arguments.get("max_results", 0)
            result = _query_fns["list_files"](pattern, max_results=max_results)

        elif name == "get_structure_summary":
            file_path = arguments.get("file_path")
            result = _query_fns["get_structure_summary"](file_path)

        elif name == "get_function_source":
            max_lines = arguments.get("max_lines", 0)
            result = _query_fns["get_function_source"](
                arguments["name"],
                arguments.get("file_path"),
                max_lines=max_lines,
            )

        elif name == "get_class_source":
            max_lines = arguments.get("max_lines", 0)
            result = _query_fns["get_class_source"](
                arguments["name"],
                arguments.get("file_path"),
                max_lines=max_lines,
            )

        elif name == "get_functions":
            file_path = arguments.get("file_path")
            max_results = arguments.get("max_results", 0)
            result = _query_fns["get_functions"](file_path, max_results=max_results)

        elif name == "get_classes":
            file_path = arguments.get("file_path")
            max_results = arguments.get("max_results", 0)
            result = _query_fns["get_classes"](file_path, max_results=max_results)

        elif name == "get_imports":
            file_path = arguments.get("file_path")
            max_results = arguments.get("max_results", 0)
            result = _query_fns["get_imports"](file_path, max_results=max_results)

        elif name == "find_symbol":
            result = _query_fns["find_symbol"](arguments["name"])

        elif name == "get_dependencies":
            max_results = arguments.get("max_results", 0)
            result = _query_fns["get_dependencies"](arguments["name"], max_results=max_results)

        elif name == "get_dependents":
            max_results = arguments.get("max_results", 0)
            result = _query_fns["get_dependents"](arguments["name"], max_results=max_results)

        elif name == "get_change_impact":
            max_direct = arguments.get("max_direct", 0)
            max_transitive = arguments.get("max_transitive", 0)
            result = _query_fns["get_change_impact"](
                arguments["name"], max_direct=max_direct, max_transitive=max_transitive
            )

        elif name == "get_call_chain":
            result = _query_fns["get_call_chain"](
                arguments["from_name"],
                arguments["to_name"],
            )

        elif name == "get_file_dependencies":
            max_results = arguments.get("max_results", 0)
            result = _query_fns["get_file_dependencies"](
                arguments["file_path"], max_results=max_results
            )

        elif name == "get_file_dependents":
            max_results = arguments.get("max_results", 0)
            result = _query_fns["get_file_dependents"](
                arguments["file_path"], max_results=max_results
            )

        elif name == "search_codebase":
            max_results = arguments.get("max_results", 100)
            result = _query_fns["search_codebase"](arguments["pattern"], max_results=max_results)

        else:
            return [TextContent(type="text", text=f"Error: unknown tool '{name}'")]

        formatted = _format_result(result)
        _total_chars_returned += len(formatted)

        # Flush stats after every call so data survives abrupt session end
        source_chars = 0
        if _indexer and _indexer._project_index:
            source_chars = sum(m.total_chars for m in _indexer._project_index.files.values())
        naive_chars = 0
        for t, c in _tool_call_counts.items():
            if t == "get_usage_stats":
                continue
            naive_chars += int(source_chars * _TOOL_COST_MULTIPLIERS.get(t, 0.10) * c)
        _flush_stats(naive_chars)

        return [TextContent(type="text", text=formatted)]

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[mcp-codebase-index] Error in {name}: {tb}", file=sys.stderr)
        return [TextContent(type="text", text=f"Error: {e}")]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main_sync():
    """Synchronous entry point for console_scripts."""
    import asyncio

    asyncio.run(main())


if __name__ == "__main__":
    main_sync()
