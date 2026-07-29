import hashlib
import os
import zipfile

import pytest

from al_medlit.core.archive import write_deterministic_zip


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
