"""Deterministic, bounded-memory archive helpers."""

from __future__ import annotations

import lzma
import shutil
import stat
import struct
import zipfile
import zlib
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Literal

_END_OF_CENTRAL_DIRECTORY_SIGNATURE = b"PK\x05\x06"
_CENTRAL_DIRECTORY_MEMBER_SIGNATURE = b"PK\x01\x02"
_END_OF_CENTRAL_DIRECTORY = struct.Struct("<4s4H2LH")
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
            not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in values
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
    end_record = None
    end_record_offset = None
    # zipfile selects the last signature, including signatures inside comments.
    # Reject ambiguous comments before it can load a different directory.
    offset = tail.rfind(_END_OF_CENTRAL_DIRECTORY_SIGNATURE)
    if offset >= 0 and offset + _END_OF_CENTRAL_DIRECTORY.size <= len(tail):
        candidate = _END_OF_CENTRAL_DIRECTORY.unpack_from(tail, offset)
        if offset + _END_OF_CENTRAL_DIRECTORY.size + candidate[7] == len(tail):
            end_record = candidate
            end_record_offset = archive_size - tail_size + offset
    if end_record is None or end_record_offset is None:
        raise ArchiveExtractionError("Archive is not a valid ZIP file")

    # Use ZipFile's bounded metadata reader so preflight checks its exact directory.
    selected = zipfile._EndRecData(source)
    if selected is None:
        raise ArchiveExtractionError("Archive is not a valid ZIP file")
    member_count = selected[zipfile._ECD_ENTRIES_TOTAL]
    central_directory_size = selected[zipfile._ECD_SIZE]
    central_directory_end = selected[zipfile._ECD_LOCATION]
    if (
        selected[zipfile._ECD_DISK_NUMBER] != 0
        or selected[zipfile._ECD_DISK_START] != 0
        or selected[zipfile._ECD_ENTRIES_THIS_DISK] != member_count
    ):
        raise ArchiveExtractionError("Archive has invalid or unsupported ZIP metadata")
    if selected[zipfile._ECD_SIGNATURE] == zipfile.stringEndArchive64:
        # Older 3.12 releases leave LOCATION at the classic EOCD, unlike newer patches.
        if central_directory_end == end_record_offset:
            central_directory_end -= zipfile.sizeEndCentDir64 + zipfile.sizeEndCentDir64Locator
    elif member_count == 0xFFFF or central_directory_size == 0xFFFFFFFF:
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
            raise ArchiveExtractionError(f"Archive contains more than {limits.max_members} members")
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
        raise ArchiveExtractionError(f"Archive contains more than {limits.max_members} members")

    validated: list[tuple[zipfile.ZipInfo, Path]] = []
    seen_paths: set[Path] = set()
    path_kinds: dict[Path, Literal["file", "directory"]] = {destination: "directory"}
    total_bytes = 0
    for member in members:
        if not member.filename or len(member.filename) > limits.max_path_length:
            raise ArchiveExtractionError("Archive contains an invalid path")
        if member.flag_bits & 1:
            raise ArchiveExtractionError("Encrypted archive members are not supported")
        if member.flag_bits & 0x60:
            raise ArchiveExtractionError("Archive contains unsupported member flags")
        zipfile._check_compression(member.compress_type)
        is_directory = member.is_dir()
        if member.create_system == 3:
            file_type = stat.S_IFMT(member.external_attr >> 16)
            expected_type = stat.S_IFDIR if is_directory else stat.S_IFREG
            if file_type not in (0, expected_type):
                raise ArchiveExtractionError("Archive contains an unsupported file type")

        path = (destination / member.filename).resolve()
        if not path.is_relative_to(destination):
            raise ArchiveExtractionError("Archive contains an unsafe path")
        if path in seen_paths:
            raise ArchiveExtractionError("Archive contains duplicate paths")
        seen_paths.add(path)
        kind = "directory" if is_directory else "file"
        if path in path_kinds and path_kinds[path] != kind:
            raise ArchiveExtractionError("Archive contains file/directory path collisions")
        path_kinds[path] = kind
        for relative_parent in path.relative_to(destination).parents:
            parent = destination / relative_parent
            if path_kinds.get(parent) == "file":
                raise ArchiveExtractionError("Archive contains file/directory path collisions")
            path_kinds[parent] = "directory"

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
    destination_path = Path(destination)
    try:
        if source_path.is_symlink() or not source_path.is_file():
            raise ArchiveExtractionError("Archive source must be a regular, non-symlink file")
        archive_size = source_path.stat().st_size
        if archive_size > extraction_limits.max_archive_bytes:
            raise ArchiveExtractionError(
                f"Archive exceeds the {extraction_limits.max_archive_bytes}-byte input limit"
            )

        destination_path.mkdir(parents=True, exist_ok=True)
        resolved_destination = destination_path.resolve()
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
                    target_opened = False
                    try:
                        with archive.open(member) as source, path.open("wb") as target:
                            target_opened = True
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
                        if target_opened:
                            with suppress(OSError):
                                path.unlink(missing_ok=True)
                        raise
    except ArchiveExtractionError:
        raise
    except zipfile.BadZipFile as exc:
        raise ArchiveExtractionError("Archive is not a valid ZIP file") from exc
    except (OSError, RuntimeError, ValueError, EOFError, zlib.error, lzma.LZMAError) as exc:
        raise ArchiveExtractionError(f"Archive could not be extracted: {exc}") from exc
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
