"""Deterministic, bounded-memory archive helpers."""

from __future__ import annotations

import shutil
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO

_END_OF_CENTRAL_DIRECTORY_SIGNATURE = b"PK\x05\x06"
_ZIP64_END_OF_CENTRAL_DIRECTORY_SIGNATURE = b"PK\x06\x06"
_ZIP64_END_OF_CENTRAL_DIRECTORY_LOCATOR_SIGNATURE = b"PK\x06\x07"
_CENTRAL_DIRECTORY_MEMBER_SIGNATURE = b"PK\x01\x02"
_END_OF_CENTRAL_DIRECTORY = struct.Struct("<4s4H2LH")
_ZIP64_END_OF_CENTRAL_DIRECTORY = struct.Struct("<4sQ2H2L4Q")
_ZIP64_END_OF_CENTRAL_DIRECTORY_LOCATOR = struct.Struct("<4sLQL")
_CENTRAL_DIRECTORY_MEMBER_SIZE = 46
_MAX_ZIP_COMMENT_BYTES = (1 << 16) - 1


@dataclass(frozen=True, slots=True)
class ArchiveExtractionLimits:
    """Hard bounds for expanding an untrusted ZIP archive."""

    max_archive_bytes: int = 16 * 1024 * 1024 * 1024
    max_central_directory_bytes: int = 128 * 1024 * 1024
    max_members: int = 512
    max_member_bytes: int = 16 * 1024 * 1024 * 1024
    max_total_bytes: int = 64 * 1024 * 1024 * 1024
    max_path_length: int = 512

    def __post_init__(self) -> None:
        values = (
            self.max_archive_bytes,
            self.max_central_directory_bytes,
            self.max_members,
            self.max_member_bytes,
            self.max_total_bytes,
            self.max_path_length,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in values
        ):
            raise ValueError("Archive extraction limits must be positive integers")


class ArchiveExtractionError(ValueError):
    """Raised when an archive cannot be extracted within the safety policy."""


def _validate_zip_inventory(source: BinaryIO, limits: ArchiveExtractionLimits) -> None:
    """Validate central-directory bounds without loading it into memory."""

    source.seek(0, 2)
    archive_size = source.tell()
    tail_size = min(
        archive_size,
        _END_OF_CENTRAL_DIRECTORY.size + _MAX_ZIP_COMMENT_BYTES,
    )
    source.seek(archive_size - tail_size)
    tail = source.read(tail_size)
    search_end = len(tail)
    end_record = None
    end_record_offset = None
    while search_end:
        offset = tail.rfind(_END_OF_CENTRAL_DIRECTORY_SIGNATURE, 0, search_end)
        if offset < 0:
            break
        if offset + _END_OF_CENTRAL_DIRECTORY.size <= len(tail):
            candidate = _END_OF_CENTRAL_DIRECTORY.unpack_from(tail, offset)
            if offset + _END_OF_CENTRAL_DIRECTORY.size + candidate[7] == len(tail):
                end_record = candidate
                end_record_offset = archive_size - tail_size + offset
                break
        search_end = offset
    if end_record is None or end_record_offset is None:
        raise ArchiveExtractionError("Archive is not a valid ZIP file")

    member_count = end_record[4]
    central_directory_size = end_record[5]
    central_directory_end = end_record_offset
    locator = None
    locator_offset = end_record_offset - _ZIP64_END_OF_CENTRAL_DIRECTORY_LOCATOR.size
    if locator_offset >= 0:
        source.seek(locator_offset)
        locator_data = source.read(_ZIP64_END_OF_CENTRAL_DIRECTORY_LOCATOR.size)
        if len(locator_data) == _ZIP64_END_OF_CENTRAL_DIRECTORY_LOCATOR.size:
            candidate = _ZIP64_END_OF_CENTRAL_DIRECTORY_LOCATOR.unpack(locator_data)
            if candidate[0] == _ZIP64_END_OF_CENTRAL_DIRECTORY_LOCATOR_SIGNATURE:
                locator = candidate
    if locator is not None:
        if locator[1] != 0 or locator[3] != 1:
            raise ArchiveExtractionError("Multi-disk ZIP archives are not supported")
        zip64_record_offset = locator_offset - _ZIP64_END_OF_CENTRAL_DIRECTORY.size
        if zip64_record_offset < 0:
            raise ArchiveExtractionError("Archive has invalid ZIP64 metadata")
        source.seek(zip64_record_offset)
        zip64_data = source.read(_ZIP64_END_OF_CENTRAL_DIRECTORY.size)
        if len(zip64_data) != _ZIP64_END_OF_CENTRAL_DIRECTORY.size:
            raise ArchiveExtractionError("Archive has truncated ZIP64 metadata")
        zip64_record = _ZIP64_END_OF_CENTRAL_DIRECTORY.unpack(zip64_data)
        if (
            zip64_record[0] != _ZIP64_END_OF_CENTRAL_DIRECTORY_SIGNATURE
            or zip64_record[1] != 44
            or zip64_record[4] != 0
            or zip64_record[5] != 0
            or zip64_record[6] != zip64_record[7]
        ):
            raise ArchiveExtractionError("Archive has invalid ZIP64 metadata")
        member_count = zip64_record[7]
        central_directory_size = zip64_record[8]
        central_directory_end = zip64_record_offset
    elif (
        member_count == 0xFFFF
        or central_directory_size == 0xFFFFFFFF
        or end_record[1] != 0
        or end_record[2] != 0
        or end_record[3] != member_count
    ):
        raise ArchiveExtractionError("Archive has invalid or unsupported ZIP metadata")

    if member_count > limits.max_members:
        raise ArchiveExtractionError(f"Archive contains more than {limits.max_members} members")
    if central_directory_size > limits.max_central_directory_bytes:
        raise ArchiveExtractionError("Archive central directory exceeds its size limit")
    central_directory_start = central_directory_end - central_directory_size
    if central_directory_start < 0:
        raise ArchiveExtractionError("Archive has invalid central-directory metadata")

    source.seek(central_directory_start)
    consumed = 0
    observed_members = 0
    while consumed < central_directory_size:
        header = source.read(_CENTRAL_DIRECTORY_MEMBER_SIZE)
        if (
            len(header) != _CENTRAL_DIRECTORY_MEMBER_SIZE
            or header[:4] != _CENTRAL_DIRECTORY_MEMBER_SIGNATURE
        ):
            raise ArchiveExtractionError("Archive has an invalid central directory")
        filename_length, extra_length, comment_length = struct.unpack_from("<3H", header, 28)
        variable_length = filename_length + extra_length + comment_length
        member_size = _CENTRAL_DIRECTORY_MEMBER_SIZE + variable_length
        if (
            filename_length == 0
            or filename_length > limits.max_path_length
            or consumed + member_size > central_directory_size
        ):
            raise ArchiveExtractionError("Archive contains an invalid path")
        observed_members += 1
        if observed_members > limits.max_members:
            raise ArchiveExtractionError(
                f"Archive contains more than {limits.max_members} members"
            )
        source.seek(variable_length, 1)
        consumed += member_size
    if observed_members != member_count:
        raise ArchiveExtractionError("Archive member count does not match its directory")


def _validated_archive_members(
    archive: zipfile.ZipFile,
    destination: Path,
    limits: ArchiveExtractionLimits,
) -> list[tuple[zipfile.ZipInfo, Path]]:
    members = archive.infolist()
    if len(members) > limits.max_members:
        raise ArchiveExtractionError(
            f"Archive contains more than {limits.max_members} members"
        )

    validated: list[tuple[zipfile.ZipInfo, Path]] = []
    seen_paths: set[Path] = set()
    total_bytes = 0
    for member in members:
        if not member.filename or len(member.filename) > limits.max_path_length:
            raise ArchiveExtractionError("Archive contains an invalid path")
        path = (destination / member.filename).resolve()
        if not path.is_relative_to(destination):
            raise ArchiveExtractionError("Archive contains an unsafe path")
        if path in seen_paths:
            raise ArchiveExtractionError("Archive contains duplicate paths")
        seen_paths.add(path)

        if member.file_size > limits.max_member_bytes:
            raise ArchiveExtractionError(
                f"Archive member exceeds the {limits.max_member_bytes}-byte limit"
            )
        total_bytes += member.file_size
        if total_bytes > limits.max_total_bytes:
            raise ArchiveExtractionError(
                f"Archive expands beyond the {limits.max_total_bytes}-byte limit"
            )
        validated.append((member, path))
    return validated


def extract_zip_bounded(
    archive_path: str | Path,
    destination: str | Path,
    *,
    limits: ArchiveExtractionLimits | None = None,
) -> Path:
    """Extract a ZIP after preflight checks and enforce limits while streaming."""

    extraction_limits = limits or ArchiveExtractionLimits()
    source_path = Path(archive_path)
    if source_path.is_symlink() or not source_path.is_file():
        raise ArchiveExtractionError("Archive source must be a regular, non-symlink file")
    archive_size = source_path.stat().st_size
    if archive_size > extraction_limits.max_archive_bytes:
        raise ArchiveExtractionError(
            f"Archive exceeds the {extraction_limits.max_archive_bytes}-byte input limit"
        )

    destination_path = Path(destination)
    destination_path.mkdir(parents=True, exist_ok=True)
    resolved_destination = destination_path.resolve()
    try:
        with source_path.open("rb") as archive_source:
            _validate_zip_inventory(archive_source, extraction_limits)

            with zipfile.ZipFile(archive_source) as archive:
                members = _validated_archive_members(
                    archive,
                    resolved_destination,
                    extraction_limits,
                )
                total_written = 0
                for member, path in members:
                    if member.is_dir():
                        path.mkdir(parents=True, exist_ok=True)
                        continue
                    path.parent.mkdir(parents=True, exist_ok=True)
                    member_written = 0
                    try:
                        with archive.open(member) as source, path.open("wb") as target:
                            while chunk := source.read(1024 * 1024):
                                member_written += len(chunk)
                                if member_written > extraction_limits.max_member_bytes:
                                    raise ArchiveExtractionError(
                                        "Archive member exceeded its extraction limit"
                                    )
                                total_written += len(chunk)
                                if total_written > extraction_limits.max_total_bytes:
                                    raise ArchiveExtractionError(
                                        "Archive exceeded its total extraction limit"
                                    )
                                target.write(chunk)
                    except Exception:
                        path.unlink(missing_ok=True)
                        raise
    except zipfile.BadZipFile as exc:
        raise ArchiveExtractionError("Archive is not a valid ZIP file") from exc
    return resolved_destination


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
