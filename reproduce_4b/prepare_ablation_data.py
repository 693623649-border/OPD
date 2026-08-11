#!/usr/bin/env python3
"""Materialize deterministic datasets used by the formal OPD ablation suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence


DOUBLE_BOX = r"\boxed{{}}"
SINGLE_BOX = r"\boxed{}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--scientific-only",
        action="store_true",
        help="Only materialize the seed-locked subsets used by the scientific suite.",
    )
    return parser


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _messages(value: Any) -> list[dict[str, Any]]:
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        value = value.tolist()
    if not isinstance(value, list) or not value:
        raise ValueError("prompt must be a non-empty message list")
    result: list[dict[str, Any]] = []
    for message in value:
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise ValueError("every prompt message must be a mapping with string content")
        result.append(dict(message))
    return result


def strict_paper_prompts(frame: Any) -> Any:
    prompts: list[list[dict[str, Any]]] = []
    replacements = 0
    for raw_prompt in frame["prompt"]:
        prompt = _messages(raw_prompt)
        content = prompt[-1]["content"]
        count = content.count(DOUBLE_BOX)
        if count != 1:
            raise ValueError(
                "released processed DAPO prompt must contain exactly one literal "
                f"{DOUBLE_BOX!r}; observed {count}"
            )
        prompt[-1]["content"] = content.replace(DOUBLE_BOX, SINGLE_BOX)
        replacements += 1
        prompts.append(prompt)
    result = frame.copy()
    result["prompt"] = prompts
    if replacements != len(result):
        raise AssertionError("not every DAPO prompt was converted")
    return result


def write_parquet_atomic(frame: Any, destination: Path, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {destination}; pass --overwrite intentionally")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def deterministic_indices(num_rows: int, size: int, seed: int) -> Any:
    """Return sorted row IDs from the suite's disclosed sampling procedure."""

    if size <= 0 or size > num_rows:
        raise ValueError(f"cannot sample {size} rows without replacement from {num_rows}")
    import numpy as np

    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(num_rows, size=size, replace=False)).astype("<i8", copy=False)


def subset_record(
    frame: Any,
    *,
    source: Path,
    destination: Path,
    size: int,
    seed: int,
    overwrite: bool,
    selected: Any | None = None,
) -> dict[str, Any]:
    """Write one deterministic subset and retain its complete row-ID provenance."""

    if selected is None:
        selected = deterministic_indices(len(frame), size, seed)
    if len(selected) != size:
        raise ValueError(f"selected {len(selected)} rows for a requested subset of {size}")
    subset = frame.iloc[selected].reset_index(drop=True)
    write_parquet_atomic(subset, destination, overwrite)
    selected_ids = [int(value) for value in selected]
    return {
        "path": str(destination),
        "rows": len(subset),
        "sha256": sha256(destination),
        "source_path": str(source),
        "source_rows": len(frame),
        "source_sha256": sha256(source),
        "sampling_method": "numpy.default_rng(seed).choice_without_replacement_then_sort",
        "seed": seed,
        "selected_row_ids": selected_ids,
        "selected_row_ids_sha256": hashlib.sha256(
            selected.astype("<i8", copy=False).tobytes()
        ).hexdigest(),
    }


def materialize_scientific_subsets(
    root: Path, seed: int, overwrite: bool
) -> dict[str, Any]:
    """Build the exact, write-once data inputs for the 2xA100 scientific matrix."""

    import pandas as pd

    output_dir = root / "datasets/ablation"
    sources = {
        "dapo_original": root / "datasets/dapo-math-17k.parquet",
        "dapo_processed": root / "datasets/dapo-math-17k-processed.parquet",
        "dapo_matched": output_dir / "dapo-paper-aligned-matched-deepmath-14116-seed42.parquet",
        "deepmath": root / "datasets/DeepMath_deduped.parquet",
        "openthoughts": root / "datasets/OpenThoughts3_opd.parquet",
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing scientific source dataset(s): " + ", ".join(missing))

    frames = {name: pd.read_parquet(path) for name, path in sources.items()}
    paper_aligned = strict_paper_prompts(frames["dapo_processed"])
    if len(frames["dapo_original"]) != len(paper_aligned):
        raise ValueError("original and paper-aligned DAPO datasets must have identical row counts")
    if len(frames["dapo_matched"]) != len(frames["deepmath"]):
        raise ValueError("matched DAPO and DeepMath datasets must have identical row counts")

    dapo_400_ids = deterministic_indices(len(frames["dapo_original"]), 400, seed)
    paired_400_ids = deterministic_indices(len(frames["deepmath"]), 400, seed)
    specs = (
        (
            "dapo_original_520",
            frames["dapo_original"],
            sources["dapo_original"],
            output_dir / f"dapo-original-520-seed{seed}.parquet",
            520,
            None,
        ),
        (
            "dapo_original_400",
            frames["dapo_original"],
            sources["dapo_original"],
            output_dir / f"dapo-original-400-seed{seed}.parquet",
            400,
            dapo_400_ids,
        ),
        (
            "dapo_paper_aligned_400",
            paper_aligned,
            sources["dapo_processed"],
            output_dir / f"dapo-paper-aligned-400-seed{seed}.parquet",
            400,
            dapo_400_ids,
        ),
        (
            "dapo_original_1200",
            frames["dapo_original"],
            sources["dapo_original"],
            output_dir / f"dapo-original-1200-seed{seed}.parquet",
            1200,
            None,
        ),
        (
            "dapo_matched_400",
            frames["dapo_matched"],
            sources["dapo_matched"],
            output_dir / f"dapo-matched-deepmath-400-seed{seed}.parquet",
            400,
            paired_400_ids,
        ),
        (
            "deepmath_400",
            frames["deepmath"],
            sources["deepmath"],
            output_dir / f"deepmath-400-seed{seed}.parquet",
            400,
            paired_400_ids,
        ),
        (
            "openthoughts_400",
            frames["openthoughts"],
            sources["openthoughts"],
            output_dir / f"openthoughts3-opd-400-seed{seed}.parquet",
            400,
            None,
        ),
    )
    subsets = {
        name: subset_record(
            frame,
            source=source_path,
            destination=destination,
            size=size,
            seed=seed,
            overwrite=overwrite,
            selected=selected,
        )
        for name, frame, source_path, destination, size, selected in specs
    }
    manifest = {
        "schema_version": 2,
        "suite_id": "rethinking-opd-scientific-2xa100-v1",
        "seed": seed,
        "subsets": subsets,
    }
    manifest_path = output_dir / f"scientific_data_manifest_seed{seed}.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {manifest_path}; pass --overwrite intentionally")
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(manifest_path)
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = sha256(manifest_path)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.repo_root.expanduser().resolve()
    source = root / "datasets/dapo-math-17k-processed.parquet"
    deepmath = root / "datasets/DeepMath_deduped.parquet"
    output_dir = root / "datasets/ablation"
    strict_output = output_dir / "dapo-math-17k-paper-aligned.parquet"

    try:
        import numpy as np
        import pandas as pd

        if args.scientific_only:
            manifest = materialize_scientific_subsets(root, args.seed, args.overwrite)
            print(json.dumps(manifest, ensure_ascii=False, indent=2))
            return 0

        if not source.is_file() or not deepmath.is_file():
            missing = [str(path) for path in (source, deepmath) if not path.is_file()]
            raise FileNotFoundError("missing source dataset(s): " + ", ".join(missing))
        dapo = pd.read_parquet(source)
        deepmath_rows = len(pd.read_parquet(deepmath, columns=["prompt"]))
        if deepmath_rows <= 0 or deepmath_rows > len(dapo):
            raise ValueError(f"cannot match {deepmath_rows} DeepMath rows from {len(dapo)} DAPO rows")

        strict = strict_paper_prompts(dapo)
        rng = np.random.default_rng(args.seed)
        selected = np.sort(rng.choice(len(strict), size=deepmath_rows, replace=False))
        matched = strict.iloc[selected].reset_index(drop=True)
        matched_output = output_dir / (
            f"dapo-paper-aligned-matched-deepmath-{deepmath_rows}-seed{args.seed}.parquet"
        )

        write_parquet_atomic(strict, strict_output, args.overwrite)
        write_parquet_atomic(matched, matched_output, args.overwrite)

        selected_digest = hashlib.sha256(selected.astype("<i8", copy=False).tobytes()).hexdigest()
        manifest = {
            "schema_version": 1,
            "seed": args.seed,
            "method": "numpy.default_rng(seed).choice_without_replacement_then_sort",
            "source": {"path": str(source), "rows": len(dapo), "sha256": sha256(source)},
            "deepmath_reference": {
                "path": str(deepmath),
                "rows": deepmath_rows,
                "sha256": sha256(deepmath),
            },
            "strict_paper_template": {
                "path": str(strict_output),
                "rows": len(strict),
                "sha256": sha256(strict_output),
                "literal_rewrite": f"{DOUBLE_BOX} -> {SINGLE_BOX}",
            },
            "matched_dapo": {
                "path": str(matched_output),
                "rows": len(matched),
                "sha256": sha256(matched_output),
                "selected_indices_sha256": selected_digest,
            },
            "paper_disclosure": (
                "The paper reports matched-size DAPO and deduplicated DeepMath but does not disclose "
                "the DAPO row IDs or sampling seed; this deterministic seed-42 subset is a local reconstruction."
            ),
        }
        manifest_path = output_dir / "ablation_data_manifest.json"
        if manifest_path.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite {manifest_path}; pass --overwrite intentionally")
        temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp-{os.getpid()}")
        with temporary_manifest.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        temporary_manifest.replace(manifest_path)
    except (FileExistsError, FileNotFoundError, ImportError, OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
