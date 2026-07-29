from scripts import runtime_preflight


def test_runtime_readiness_fails_closed_without_an_image_digest(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        runtime_preflight.importlib.util,
        "find_spec",
        lambda _name: object(),
    )
    monkeypatch.setattr(
        runtime_preflight.importlib.metadata,
        "version",
        lambda _distribution: "1.0.0",
    )
    monkeypatch.setattr(
        runtime_preflight,
        "_probe_device",
        lambda _required: ("cpu", True, 16 * 1024**3),
    )

    unattested = runtime_preflight.collect_runtime_readiness(
        "classical-cpu",
        scratch_root=tmp_path,
        worker_image_digest=None,
        storage_probe=lambda: True,
    )
    attested = runtime_preflight.collect_runtime_readiness(
        "classical-cpu",
        scratch_root=tmp_path,
        worker_image_digest="a" * 64,
        storage_probe=lambda: True,
    )

    assert unattested.ready is False
    assert attested.ready is True


def test_worker_image_digest_normalization_treats_compose_blanks_as_missing():
    assert runtime_preflight.normalize_worker_image_digest(None) is None
    assert runtime_preflight.normalize_worker_image_digest("") is None
    assert runtime_preflight.normalize_worker_image_digest("   ") is None
    assert runtime_preflight.normalize_worker_image_digest(
        " sha256:" + "a" * 64 + " "
    ) == "a" * 64
