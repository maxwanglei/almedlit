import hashlib
import os
import stat
import struct
import zipfile
import zlib
from pathlib import Path

import pytest

from al_medlit.core.archive import (
    ArchiveExtractionError,
    ArchiveExtractionLimits,
    extract_zip_bounded,
    write_deterministic_zip,
)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def test_deterministic_zip_streams_files_and_normalizes_metadata(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    payload = root / "large-shard.safetensors"
    payload.write_bytes(b"model-shard" * 200_000)
    config = root / "config.json"
    config.write_text('{"model":"fixture"}\n', encoding="utf-8")
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    def reject_read_bytes(_self):
        raise AssertionError("archive creation must not load a whole file with read_bytes")

    monkeypatch.setattr("pathlib.Path.read_bytes", reject_read_bytes)
    write_deterministic_zip(first, root, [payload.name, config.name])
    os.utime(payload, (1_700_000_000, 1_700_000_000))
    payload.chmod(0o600)
    write_deterministic_zip(second, root, [config.name, payload.name])

    assert _sha256(first) == _sha256(second)
    with zipfile.ZipFile(first) as archive:
        assert archive.namelist() == ["config.json", "large-shard.safetensors"]
        for info in archive.infolist():
            assert info.date_time == (1980, 1, 1, 0, 0, 0)
            assert info.external_attr >> 16 == 0o100644


def test_deterministic_zip_rejects_paths_outside_the_root(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (tmp_path / "secret").write_text("not an archive input", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsafe archive path"):
        write_deterministic_zip(tmp_path / "unsafe.zip", root, ["../secret"])

    link = root / "linked-secret"
    link.symlink_to(tmp_path / "secret")
    with pytest.raises(ValueError, match="source is a symlink"):
        write_deterministic_zip(tmp_path / "symlink.zip", root, [link.name])


def _write_test_zip(path, members):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members:
            archive.writestr(name, payload)


def test_bounded_extraction_accepts_archive_within_limits(tmp_path):
    archive_path = tmp_path / "checkpoint.zip"
    _write_test_zip(
        archive_path,
        [
            ("checkpoint/config.json", b"{}\n"),
            ("checkpoint/model.safetensors", b"weights"),
        ],
    )

    destination = extract_zip_bounded(
        archive_path,
        tmp_path / "extracted",
        limits=ArchiveExtractionLimits(
            max_archive_bytes=1024,
            max_members=2,
            max_member_bytes=16,
            max_total_bytes=16,
            max_path_length=64,
        ),
    )

    assert (destination / "checkpoint/config.json").read_bytes() == b"{}\n"
    assert (destination / "checkpoint/model.safetensors").read_bytes() == b"weights"


@pytest.mark.parametrize(
    ("members", "limit_overrides", "message"),
    [
        (
            [("checkpoint/first", b"1"), ("checkpoint/second", b"2")],
            {"max_members": 1},
            "more than 1 members",
        ),
        (
            [("checkpoint/model", b"12345")],
            {"max_member_bytes": 4},
            "member exceeds the 4-byte limit",
        ),
        (
            [("checkpoint/first", b"123"), ("checkpoint/second", b"456")],
            {"max_total_bytes": 5},
            "expands beyond the 5-byte limit",
        ),
    ],
)
def test_bounded_extraction_rejects_resource_limit_violations(
    tmp_path,
    members,
    limit_overrides,
    message,
):
    archive_path = tmp_path / "checkpoint.zip"
    _write_test_zip(archive_path, members)
    destination = tmp_path / "extracted"
    limit_values = {
        "max_archive_bytes": 1024,
        "max_members": 10,
        "max_member_bytes": 10,
        "max_total_bytes": 20,
        "max_path_length": 64,
        **limit_overrides,
    }
    limits = ArchiveExtractionLimits(
        **limit_values,
    )

    with pytest.raises(ArchiveExtractionError, match=message):
        extract_zip_bounded(archive_path, destination, limits=limits)

    assert not any(destination.rglob("*"))


def test_bounded_extraction_rejects_archive_input_size_before_opening(tmp_path):
    archive_path = tmp_path / "checkpoint.zip"
    _write_test_zip(archive_path, [("checkpoint/model", b"payload")])

    with pytest.raises(ArchiveExtractionError, match="input limit"):
        extract_zip_bounded(
            archive_path,
            tmp_path / "extracted",
            limits=ArchiveExtractionLimits(max_archive_bytes=archive_path.stat().st_size - 1),
        )


def test_bounded_extraction_checks_member_count_before_loading_directory(
    tmp_path,
    monkeypatch,
):
    archive_path = tmp_path / "checkpoint.zip"
    _write_test_zip(
        archive_path,
        [("checkpoint/first", b"1"), ("checkpoint/second", b"2")],
    )

    def reject_zipfile_open(*_args, **_kwargs):
        raise AssertionError("oversized inventory must be rejected before ZipFile opens")

    monkeypatch.setattr("al_medlit.core.archive.zipfile.ZipFile", reject_zipfile_open)
    with pytest.raises(ArchiveExtractionError, match="more than 1 members"):
        extract_zip_bounded(
            archive_path,
            tmp_path / "extracted",
            limits=ArchiveExtractionLimits(max_members=1),
        )


def test_bounded_extraction_rejects_traversal_before_writing(tmp_path):
    archive_path = tmp_path / "checkpoint.zip"
    _write_test_zip(
        archive_path,
        [("checkpoint/config.json", b"{}"), ("../outside", b"escaped")],
    )
    destination = tmp_path / "extracted"

    with pytest.raises(ArchiveExtractionError, match="unsafe path"):
        extract_zip_bounded(archive_path, destination)

    assert not any(destination.rglob("*"))
    assert not (tmp_path / "outside").exists()


def test_checkpoint_extractors_surface_resource_limit_errors(tmp_path):
    from al_medlit.core.exceptions import ConflictError
    from al_medlit.inference.execution import _extract_checkpoint
    from al_medlit.training.runner import RunnerError, _extract_zip

    archive_path = tmp_path / "checkpoint.zip"
    _write_test_zip(archive_path, [("checkpoint/model", b"12345")])
    limits = ArchiveExtractionLimits(max_member_bytes=4)

    with pytest.raises(ConflictError, match="member exceeds the 4-byte limit"):
        _extract_checkpoint(archive_path, tmp_path / "local-extracted", limits=limits)
    with pytest.raises(RunnerError, match="member exceeds the 4-byte limit"):
        _extract_zip(archive_path, tmp_path / "remote-extracted", limits=limits)


def _write_zip64_test_zip(
    path, *, extension=b"", disks=1, prefix=b"", decoy=False, reverse_decoy=False
):
    _write_test_zip(path, [("checkpoint/one", b"1"), ("checkpoint/two", b"2")])
    data = path.read_bytes()
    end = struct.unpack("<4s4H2LH", data[-22:])
    directory_size, directory_offset = end[5:7]
    directory = data[directory_offset : directory_offset + directory_size]
    contents = data[:directory_offset]
    member_count = 2
    if decoy:
        entry_size = 46 + sum(struct.unpack_from("<3H", directory, 28))
        decoy_directory = directory if reverse_decoy else directory[:entry_size]
        if reverse_decoy:
            directory = directory[:entry_size]
            member_count = 1
        directory_size = len(directory)
        record_offset = len(contents) + directory_size
        extension = decoy_directory + struct.pack(
            "<4sQ2H2L4Q",
            b"PK\x06\x06",
            44,
            45,
            45,
            0,
            0,
            2 if reverse_decoy else 1,
            2 if reverse_decoy else 1,
            len(decoy_directory),
            record_offset + 56,
        )
    record_offset = len(contents) + directory_size
    record = struct.pack(
        "<4sQ2H2L4Q",
        b"PK\x06\x06",
        44 + len(extension),
        45,
        45,
        0,
        0,
        member_count,
        member_count,
        directory_size,
        directory_offset,
    )
    locator = struct.pack("<4sLQL", b"PK\x06\x07", 0, record_offset, disks)
    path.write_bytes(prefix + contents + directory + record + extension + locator + data[-22:])
    return len(decoy_directory) if reverse_decoy else directory_size


@pytest.mark.parametrize("limit_kind", ["members", "directory_bytes"])
def test_zip64_decoy_cannot_bypass_inventory_limits(tmp_path, monkeypatch, limit_kind):
    archive_path = tmp_path / "decoy.zip"
    directory_size = _write_zip64_test_zip(archive_path, decoy=True)
    with zipfile.ZipFile(archive_path) as archive:
        uses_legacy_selector = len(archive.infolist()) == 1
    if uses_legacy_selector:
        directory_size = _write_zip64_test_zip(archive_path, decoy=True, reverse_decoy=True)
    limits = (
        ArchiveExtractionLimits(max_members=1)
        if limit_kind == "members"
        else ArchiveExtractionLimits(max_central_directory_bytes=directory_size - 1)
    )
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.namelist() == ["checkpoint/one", "checkpoint/two"]

    def reject_directory_loading(*_args, **_kwargs):
        raise AssertionError("ZIP64 inventory limit must be checked before ZipFile opens")

    monkeypatch.setattr("al_medlit.core.archive.zipfile.ZipFile", reject_directory_loading)
    with pytest.raises(ArchiveExtractionError):
        extract_zip_bounded(archive_path, tmp_path / "extracted", limits=limits)


@pytest.mark.parametrize(
    ("extension", "disks", "prefix"),
    [
        (b"", 0, b""),
        (b"", 1, b""),
        (b"extended metadata", 1, b""),
        (b"", 1, b"prepended executable data\n"),
    ],
)
def test_bounded_extraction_accepts_zip64_metadata_variants(tmp_path, extension, disks, prefix):
    archive_path = tmp_path / "zip64.zip"
    _write_zip64_test_zip(archive_path, extension=extension, disks=disks, prefix=prefix)
    if extension:
        try:
            with zipfile.ZipFile(archive_path):
                pass
        except zipfile.BadZipFile:
            pytest.skip("This Python ZIP reader does not support ZIP64 extensible data")

    destination = extract_zip_bounded(archive_path, tmp_path / "extracted")

    assert (destination / "checkpoint/one").read_bytes() == b"1"
    assert (destination / "checkpoint/two").read_bytes() == b"2"


@pytest.mark.parametrize(
    "corruption", ["locator_offset", "record_size", "directory_offset", "disks"]
)
def test_bounded_extraction_rejects_invalid_zip64_metadata(tmp_path, monkeypatch, corruption):
    archive_path = tmp_path / "invalid.zip"
    _write_zip64_test_zip(archive_path)
    data = bytearray(archive_path.read_bytes())
    locator_offset = len(data) - 42
    record_offset = locator_offset - 56
    if corruption == "locator_offset":
        struct.pack_into("<Q", data, locator_offset + 8, locator_offset)
    elif corruption == "record_size":
        struct.pack_into("<Q", data, record_offset + 4, 45)
    elif corruption == "directory_offset":
        struct.pack_into("<Q", data, record_offset + 48, 0)
    else:
        struct.pack_into("<L", data, locator_offset + 16, 2)
    archive_path.write_bytes(data)
    try:
        with zipfile.ZipFile(archive_path):
            pass
    except zipfile.BadZipFile:
        pass
    else:
        pytest.skip("This Python ZIP reader accepts the malformed ZIP64 metadata")

    def reject_directory_loading(*_args, **_kwargs):
        raise AssertionError("invalid ZIP64 metadata must be rejected before ZipFile opens")

    monkeypatch.setattr("al_medlit.core.archive.zipfile.ZipFile", reject_directory_loading)
    with pytest.raises(ArchiveExtractionError):
        extract_zip_bounded(archive_path, tmp_path / "extracted")


def _write_unsupported_test_zip(path, kind):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("first", b"safe")
        member = zipfile.ZipInfo("unsupported")
        member.create_system = 3
        if kind == "symlink":
            member.external_attr = (stat.S_IFLNK | 0o777) << 16
        elif kind == "fifo":
            member.external_attr = (stat.S_IFIFO | 0o600) << 16
        archive.writestr(member, b"target")
    if kind in {"encrypted", "compression"}:
        data = bytearray(path.read_bytes())
        local = data.rfind(b"PK\x03\x04")
        central = data.rfind(b"PK\x01\x02")
        if kind == "encrypted":
            struct.pack_into("<H", data, local + 6, 1)
            struct.pack_into("<H", data, central + 8, 1)
        else:
            struct.pack_into("<H", data, local + 8, 99)
            struct.pack_into("<H", data, central + 10, 99)
        path.write_bytes(data)


@pytest.mark.parametrize("kind", ["encrypted", "compression", "symlink", "fifo"])
def test_bounded_extraction_rejects_unsupported_entries_before_writing(tmp_path, kind):
    archive_path = tmp_path / "unsupported.zip"
    _write_unsupported_test_zip(archive_path, kind)
    destination = tmp_path / "extracted"

    with pytest.raises(ArchiveExtractionError):
        extract_zip_bounded(archive_path, destination)

    assert not any(destination.rglob("*"))


@pytest.mark.parametrize(
    "names",
    [("foo", "foo/bar"), ("foo/bar", "foo"), ("foo", "foo/")],
)
def test_bounded_extraction_rejects_file_directory_collisions_before_writing(tmp_path, names):
    archive_path = tmp_path / "collision.zip"
    _write_test_zip(
        archive_path,
        [(name, b"data" if not name.endswith("/") else b"") for name in names],
    )
    destination = tmp_path / "extracted"

    with pytest.raises(ArchiveExtractionError):
        extract_zip_bounded(archive_path, destination)

    assert not any(destination.rglob("*"))


@pytest.mark.parametrize("names", [("foo/", "foo/bar"), ("foo/bar", "foo/")])
def test_bounded_extraction_accepts_explicit_parent_directories(tmp_path, names):
    archive_path = tmp_path / "directories.zip"
    _write_test_zip(
        archive_path,
        [(name, b"data" if not name.endswith("/") else b"") for name in names],
    )

    destination = extract_zip_bounded(archive_path, tmp_path / "extracted")

    assert (destination / "foo/bar").read_bytes() == b"data"


@pytest.mark.parametrize("existing_kind", ["file_parent", "directory_target"])
def test_bounded_extraction_wraps_existing_destination_collisions(tmp_path, existing_kind):
    archive_path = tmp_path / "collision.zip"
    destination = tmp_path / "extracted"
    destination.mkdir()
    if existing_kind == "file_parent":
        existing = destination / "foo"
        existing.write_bytes(b"keep")
        name = "foo/bar"
    else:
        existing = destination / "foo"
        existing.mkdir()
        (existing / "keep").write_bytes(b"keep")
        name = "foo"
    _write_test_zip(archive_path, [(name, b"replacement")])

    with pytest.raises(ArchiveExtractionError):
        extract_zip_bounded(archive_path, destination)

    preserved = existing if existing_kind == "file_parent" else existing / "keep"
    assert preserved.read_bytes() == b"keep"


def test_bounded_extraction_preserves_existing_file_when_archive_open_fails(tmp_path):
    archive_path = tmp_path / "encrypted.zip"
    _write_unsupported_test_zip(archive_path, "encrypted")
    destination = tmp_path / "extracted"
    destination.mkdir()
    existing = destination / "unsupported"
    existing.write_bytes(b"keep")

    with pytest.raises(ArchiveExtractionError):
        extract_zip_bounded(archive_path, destination)

    assert existing.read_bytes() == b"keep"


@pytest.mark.parametrize("kind", ["encrypted", "compression", "symlink", "fifo"])
def test_checkpoint_extractors_surface_unsupported_archive_errors(tmp_path, kind):
    from al_medlit.core.exceptions import ConflictError
    from al_medlit.inference.execution import _extract_checkpoint
    from al_medlit.training.runner import RunnerError, _extract_zip

    archive_path = tmp_path / "unsupported.zip"
    _write_unsupported_test_zip(archive_path, kind)

    with pytest.raises(ConflictError, match="Unsafe checkpoint archive"):
        _extract_checkpoint(archive_path, tmp_path / "local-extracted")
    with pytest.raises(RunnerError, match="Unsafe checkpoint archive"):
        _extract_zip(archive_path, tmp_path / "remote-extracted")


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_bounded_extraction_wraps_corrupt_deflate_and_preserves_cause(
    tmp_path, monkeypatch, cleanup_fails
):
    archive_path = tmp_path / "corrupt.zip"
    _write_test_zip(archive_path, [("broken", b"model weights" * 100)])
    data = bytearray(archive_path.read_bytes())
    filename_length, extra_length = struct.unpack_from("<2H", data, 26)
    payload_offset = 30 + filename_length + extra_length
    data[payload_offset : payload_offset + 2] = b"\x06\x00"
    archive_path.write_bytes(data)
    destination = tmp_path / "extracted"
    original_unlink = Path.unlink

    def unlink(path, *args, **kwargs):
        if cleanup_fails and path == destination / "broken":
            raise PermissionError("cleanup denied")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)
    with pytest.raises(ArchiveExtractionError) as caught:
        extract_zip_bounded(archive_path, destination)

    assert isinstance(caught.value.__cause__, zlib.error)
    assert (destination / "broken").exists() is cleanup_fails


def test_bounded_extraction_preserves_existing_file_when_target_open_fails(tmp_path, monkeypatch):
    archive_path = tmp_path / "checkpoint.zip"
    _write_test_zip(archive_path, [("model", b"replacement")])
    destination = tmp_path / "extracted"
    destination.mkdir()
    existing = destination / "model"
    existing.write_bytes(b"keep")
    original_open = Path.open

    def open_target(path, mode="r", *args, **kwargs):
        if path == existing and mode == "wb":
            raise PermissionError("target denied")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_target)
    with pytest.raises(ArchiveExtractionError) as caught:
        extract_zip_bounded(archive_path, destination)

    assert isinstance(caught.value.__cause__, PermissionError)
    assert existing.read_bytes() == b"keep"
