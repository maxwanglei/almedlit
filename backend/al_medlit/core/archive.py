"""Deterministic, bounded-memory archive helpers."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path, PurePosixPath


class _DeterministicZipFile(zipfile.ZipFile):
    """A ``ZipFile`` whose ``write`` method normalizes file metadata.

    ``ZipFile.writestr(path.read_bytes())`` briefly holds each complete model
    shard in memory.  This implementation keeps the familiar ``write`` API but
    streams source files in bounded chunks through ``ZipFile.open``.
    """

    def write(  # type: ignore[override]
        self,
        filename: str | Path,
        arcname: str | None = None,
        compress_type: int | None = None,
        compresslevel: int | None = None,
    ) -> None:
        source_path = Path(filename)
        if source_path.is_symlink() or not source_path.is_file():
            raise ValueError(f"Archive source is not a regular file: {source_path}")
        normalized_name = PurePosixPath(arcname or source_path.name).as_posix()
        info = zipfile.ZipInfo(normalized_name, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = self.compression if compress_type is None else compress_type
        info.external_attr = 0o100644 << 16
        info.create_system = 3
        if compresslevel is not None:
            info._compresslevel = compresslevel  # noqa: SLF001 - mirrors ZipFile.write
        with source_path.open("rb") as source, self.open(
            info,
            mode="w",
            force_zip64=True,
        ) as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)


def write_deterministic_zip(
    destination: str | Path,
    root: str | Path,
    relative_paths: list[str] | tuple[str, ...],
) -> None:
    """Write regular files under ``root`` in a reproducible streaming ZIP."""

    root_path = Path(root).resolve()
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with _DeterministicZipFile(
        destination_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        seen: set[str] = set()
        for raw_relative in sorted(relative_paths):
            relative = PurePosixPath(raw_relative)
            normalized = relative.as_posix()
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or not relative.parts
                or "\\" in normalized
            ):
                raise ValueError(f"Unsafe archive path: {raw_relative!r}")
            if normalized in seen:
                raise ValueError(f"Duplicate archive path: {raw_relative!r}")
            seen.add(normalized)
            unresolved_source = root_path
            for part in relative.parts:
                unresolved_source /= part
                if unresolved_source.is_symlink():
                    raise ValueError(f"Archive source is a symlink: {raw_relative!r}")
            source = unresolved_source.resolve()
            if not source.is_relative_to(root_path):
                raise ValueError(f"Unsafe archive source: {raw_relative!r}")
            archive.write(source, arcname=normalized)
