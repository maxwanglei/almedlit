import hashlib
import os
import zipfile

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
            ("checkpoint/config.json", b'{}\n'),
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
