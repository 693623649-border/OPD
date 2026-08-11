#!/usr/bin/env python3
"""Resolve mutable Hub IDs to immutable local snapshots for one OPD run."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Sequence

import requests
from huggingface_hub import HfApi, snapshot_download


MODEL_PATTERNS = (
    "*.json",
    "*.jinja",
    "*.model",
    "*.py",
    "*.safetensors",
    "*.txt",
)
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def is_hub_network_error(error: BaseException) -> bool:
    """Recognize transport/offline failures without treating auth or 404 as offline."""

    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(
            current,
            (
                ConnectionError,
                TimeoutError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
            ),
        ):
            return True
        response = getattr(current, "response", None)
        status_code = getattr(response, "status_code", None)
        if isinstance(status_code, int) and status_code >= 500:
            return True
        current = current.__cause__ or current.__context__
    return False


def _offline_mode() -> bool:
    """Report whether the offline contract is active for Hub resolution."""

    return os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get(
        "TRANSFORMERS_OFFLINE"
    ) == "1"


def resolve_hub_revision(source: str, revision: str) -> tuple[str, bool]:
    """Resolve a Hub reference and report whether only an exact cache may be used."""

    if _offline_mode():
        # The offline contract forbids Hub model_info calls entirely.  Only an
        # immutable caller-supplied commit can be recovered from the local cache.
        if COMMIT_RE.fullmatch(revision):
            return revision, True
        raise RuntimeError(
            f"offline contract forbids Hub resolution of non-immutable revision "
            f"{source}@{revision!r}"
        )
    requested = None if revision == "auto" else revision
    try:
        info = HfApi().model_info(source, revision=requested)
    except Exception as exc:
        # A caller-supplied full commit is already immutable.  It is the only
        # revision that may be recovered from cache when the Hub is offline.
        if COMMIT_RE.fullmatch(revision) and is_hub_network_error(exc):
            return revision, True
        raise
    resolved = str(info.sha)
    if not COMMIT_RE.fullmatch(resolved):
        raise ValueError(f"Hub did not resolve {source}@{revision} to a full commit SHA: {resolved!r}")
    if COMMIT_RE.fullmatch(revision) and resolved != revision:
        raise ValueError(
            f"Hub resolved fixed revision {source}@{revision} to unexpected commit {resolved}"
        )
    return resolved, False


def download_fixed_snapshot(
    source: str, resolved_revision: str, *, cache_only: bool
) -> tuple[Path, bool]:
    """Download one immutable snapshot, with an explicit exact-cache fallback."""

    if not COMMIT_RE.fullmatch(resolved_revision):
        raise ValueError(f"snapshot revision must be a full commit SHA: {resolved_revision!r}")
    kwargs = {
        "repo_id": source,
        "revision": resolved_revision,
        "allow_patterns": list(MODEL_PATTERNS),
    }
    if cache_only:
        try:
            return Path(snapshot_download(**kwargs, local_files_only=True)).resolve(), True
        except Exception as exc:
            raise RuntimeError(
                f"Hub is unavailable and fixed revision {source}@{resolved_revision} "
                "is not complete in the local cache"
            ) from exc
    try:
        return Path(snapshot_download(**kwargs)).resolve(), False
    except Exception as exc:
        if not is_hub_network_error(exc):
            raise
        try:
            snapshot = Path(snapshot_download(**kwargs, local_files_only=True)).resolve()
        except Exception as cache_exc:
            raise RuntimeError(
                f"Hub download failed and fixed revision {source}@{resolved_revision} "
                "is not complete in the local cache"
            ) from cache_exc
        return snapshot, True


def validate_snapshot(snapshot: Path) -> None:
    """Reject incomplete model snapshots, including partial sharded caches."""

    if not (snapshot / "config.json").is_file():
        raise ValueError(f"model snapshot has no config.json: {snapshot}")
    safetensors = list(snapshot.glob("*.safetensors"))
    indexes = list(snapshot.glob("*.safetensors.index.json"))
    if not safetensors:
        raise ValueError(f"model snapshot has no safetensors weights: {snapshot}")
    for index_path in indexes:
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read safetensors index {index_path}: {exc}") from exc
        weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"safetensors index has no non-empty weight_map: {index_path}")
        required_shards = {str(name) for name in weight_map.values()}
        missing = sorted(name for name in required_shards if not (snapshot / name).is_file())
        if missing:
            raise ValueError(
                f"model snapshot is missing {len(missing)} indexed safetensors shard(s): "
                + ", ".join(missing[:3])
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", required=True)
    parser.add_argument("--student-revision", default="auto")
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--teacher-revision", default="auto")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def pin_one(source: str, revision: str) -> dict[str, Any]:
    local = Path(source).expanduser()
    if local.exists():
        snapshot = local.resolve()
        if not snapshot.is_dir():
            raise ValueError(f"model path is not a directory: {snapshot}")
        resolved_revision = "local"
        cache_fallback = False
    else:
        resolved_revision, cache_only = resolve_hub_revision(source, revision)
        snapshot, cache_fallback = download_fixed_snapshot(
            source, resolved_revision, cache_only=cache_only
        )
        if COMMIT_RE.fullmatch(snapshot.name) and snapshot.name != resolved_revision:
            raise ValueError(
                f"snapshot path commit {snapshot.name} differs from resolved revision {resolved_revision}"
            )

    validate_snapshot(snapshot)
    return {
        "source": source,
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
        "snapshot_path": str(snapshot),
        "cache_fallback": cache_fallback,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = {
        "student": pin_one(args.student, args.student_revision),
        "teacher": pin_one(args.teacher, args.teacher_revision),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    except FileExistsError as exc:
        raise SystemExit(f"refusing to overwrite snapshot manifest: {output}") from exc
    print(f"Model snapshot manifest: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
