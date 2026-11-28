"""Dataset Downloader: Reproducible benchmark acquisition with checksum & Zip-Slip guards."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional
import urllib.request
import yaml
import zipfile


def sha256_file(path: Path) -> str:
    """Computes uppercase hexadecimal SHA-256 checksum of a file cleanly in chunks."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest().upper()


def verify_checksum(path: Path, expected_sha256: str) -> bool:
    """Verifies file integrity against a pinned SHA-256 digest."""
    actual = sha256_file(path)
    return actual.upper() == expected_sha256.strip().upper()


def safe_extract_zip(archive_path: Path, target_dir: Path, overwrite: bool = False) -> List[Path]:
    """Extracts a zip archive while rejecting path-traversal (Zip-Slip) entries."""
    target_root = target_dir.resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    extracted: List[Path] = []

    with zipfile.ZipFile(archive_path, "r") as zf:
        for info in zf.infolist():
            member_path = (target_root / info.filename).resolve()
            try:
                member_path.relative_to(target_root)
            except ValueError as exc:
                raise ValueError(
                    f"Security violation: unsafe zip entry '{info.filename}' escapes {target_root}"
                ) from exc

        for info in zf.infolist():
            member_path = (target_root / info.filename).resolve()
            if info.is_dir():
                member_path.mkdir(parents=True, exist_ok=True)
                continue
            member_path.parent.mkdir(parents=True, exist_ok=True)
            if member_path.exists() and not overwrite:
                extracted.append(member_path)
                continue
            with zf.open(info, "r") as src, open(member_path, "wb") as dst:
                while chunk := src.read(65536):
                    dst.write(chunk)
            extracted.append(member_path)

    return extracted


def _default_url_download(url: str, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, tmp_path)
        tmp_path.replace(dest_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def acquire_dataset_archive(
    urls: Iterable[str],
    dest_path: Path,
    expected_sha256: Optional[str] = None,
    force: bool = False,
    fetcher: Callable[[str, Path], None] = _default_url_download,
) -> Dict[str, Any]:
    """Downloads or verifies an existing pinned archive without silently replacing valid files."""
    dest = Path(dest_path)
    if dest.exists():
        actual_sha = sha256_file(dest)
        if expected_sha256 and actual_sha != expected_sha256.strip().upper():
            if not force:
                raise ValueError(
                    f"Existing archive {dest} failed SHA-256 verification "
                    f"(expected {expected_sha256}, got {actual_sha}). Pass force=True to re-download."
                )
        elif not force:
            return {
                "path": str(dest),
                "source_url": "local_verified_cache",
                "sha256": actual_sha,
                "size_bytes": dest.stat().st_size,
                "status": "cached_verified",
            }

    url_list = [u for u in urls if u]
    if not url_list:
        raise ValueError("No download URLs provided and archive is missing.")

    last_err: Optional[Exception] = None
    used_url: Optional[str] = None
    for url in url_list:
        try:
            fetcher(url, dest)
            used_url = url
            break
        except Exception as exc:
            last_err = exc

    if used_url is None:
        raise RuntimeError(f"All dataset download sources failed: {last_err}") from last_err

    actual_sha = sha256_file(dest)
    if expected_sha256 and actual_sha != expected_sha256.strip().upper():
        raise ValueError(
            f"Downloaded archive {dest} failed SHA-256 verification: "
            f"expected {expected_sha256}, got {actual_sha}"
        )

    return {
        "path": str(dest),
        "source_url": used_url,
        "sha256": actual_sha,
        "size_bytes": dest.stat().st_size,
        "status": "downloaded_verified",
    }


def download_from_config(
    config_path: str | Path = "configs/data.yaml",
    force: bool = False,
    extract: bool = True,
) -> Dict[str, Any]:
    """Downloads configured datasets, verifies checksums, extracts safely, and writes metadata."""
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    raw_dir = Path(cfg.get("raw_dir", "./data/raw"))
    spider_cfg = cfg.get("datasets", {}).get("spider", {})
    urls: List[str] = []
    if spider_cfg.get("source_url"):
        urls.append(spider_cfg["source_url"])
    urls.extend(spider_cfg.get("fallback_mirrors", []))

    archive_path = raw_dir / "spider.zip"
    expected_sha = spider_cfg.get("sha256")
    record = acquire_dataset_archive(
        urls=urls,
        dest_path=archive_path,
        expected_sha256=expected_sha,
        force=force,
    )

    if extract and archive_path.exists():
        safe_extract_zip(archive_path, raw_dir / "spider", overwrite=force)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "spider": record,
    }
    manifest_path = raw_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire and verify benchmark datasets.")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-extract", action="store_true")
    args = parser.parse_args()
    out = download_from_config(args.config, force=args.force, extract=not args.no_extract)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
