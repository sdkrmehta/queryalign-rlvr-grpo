"""Database-Disjoint Split Manifests: Reproducible train/val/test splitting without leakage."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Tuple


def split_records_disjoint_by_db(
    records: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 3407,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Splits records at the database level so zero DBs cross between train/val/test."""
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        db_id = rec.get("db_id")
        if not db_id:
            raise ValueError("Cannot perform DB-disjoint split: record missing 'db_id'")
        grouped[str(db_id)].append(rec)

    db_ids = sorted(grouped.keys())
    rng = random.Random(seed)
    rng.shuffle(db_ids)

    n_dbs = len(db_ids)
    n_train = max(1, int(round(n_dbs * train_ratio))) if n_dbs >= 3 else max(1, n_dbs - 2)
    n_val = max(1, int(round(n_dbs * val_ratio))) if n_dbs >= 3 else (1 if n_dbs >= 2 else 0)
    if n_train + n_val >= n_dbs and n_dbs >= 3:
        n_train = max(1, n_dbs - 2)
        n_val = 1

    train_dbs = set(db_ids[:n_train])
    val_dbs = set(db_ids[n_train : n_train + n_val])
    test_dbs = set(db_ids[n_train + n_val :])

    assert not (train_dbs & val_dbs)
    assert not (train_dbs & test_dbs)
    assert not (val_dbs & test_dbs)

    split_id = f"split_v1_seed_{seed}"
    train_out: List[Dict[str, Any]] = []
    val_out: List[Dict[str, Any]] = []
    test_out: List[Dict[str, Any]] = []

    for db in db_ids:
        target_split, target_list = (
            ("train", train_out)
            if db in train_dbs
            else ("val", val_out)
            if db in val_dbs
            else ("test", test_out)
        )
        for rec in grouped[db]:
            copy_rec = dict(rec)
            copy_rec["split"] = target_split
            copy_rec["split_manifest_id"] = split_id
            target_list.append(copy_rec)

    return train_out, val_out, test_out


def verify_no_db_leakage(
    train_records: List[Dict[str, Any]],
    *other_splits: List[Dict[str, Any]],
) -> None:
    """Raises ValueError if any database ID overlaps between train and evaluation splits."""
    train_dbs = {str(r["db_id"]).lower() for r in train_records if r.get("db_id")}
    for idx, split in enumerate(other_splits, start=1):
        other_dbs = {str(r["db_id"]).lower() for r in split if r.get("db_id")}
        overlap = sorted(train_dbs & other_dbs)
        if overlap:
            raise ValueError(
                f"Database-level split leakage detected against split #{idx}: {overlap}"
            )


def write_immutable_manifests(
    output_dir: Path,
    splits: Dict[str, List[Dict[str, Any]]],
    overwrite: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Writes immutable JSONL split manifests and returns SHA-256 + row/DB metadata."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metadata: Dict[str, Dict[str, Any]] = {}

    if "train" in splits:
        others = [rows for name, rows in splits.items() if name != "train"]
        verify_no_db_leakage(splits["train"], *others)

    for split_name, rows in splits.items():
        manifest_path = out / f"{split_name}_manifest.jsonl"
        if manifest_path.exists() and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite frozen manifest {manifest_path}; pass overwrite=True explicitly."
            )
        payload = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
        if payload:
            payload += "\n"
        manifest_path.write_text(payload, encoding="utf-8")
        digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest().upper()
        metadata[split_name] = {
            "path": str(manifest_path),
            "row_count": len(rows),
            "db_count": len({r["db_id"] for r in rows if r.get("db_id")}),
            "sha256": digest,
        }

    meta_path = out / "split_checksums.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
