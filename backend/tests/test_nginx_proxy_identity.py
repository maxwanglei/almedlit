import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "frontend/nginx/19-trusted-proxies.sh"


def _render(tmp_path, networks):
    return subprocess.run(
        ["sh", str(SCRIPT)],
        env={**os.environ, "NGINX_ENVSUBST_OUTPUT_DIR": str(tmp_path),
             "AL_MEDLIT_TRUSTED_PROXY_CIDRS": networks},
        capture_output=True, text=True, check=False,
    )


def test_direct_deployment_does_not_trust_forwarded_client_addresses(tmp_path):
    assert _render(tmp_path, "").returncode == 0
    assert "set_real_ip_from" not in (tmp_path / "01-trusted-proxies.conf").read_text()


def test_only_explicit_proxy_networks_can_supply_client_identity(tmp_path):
    assert _render(tmp_path, "172.18.0.1/32 2001:db8::1/128").returncode == 0
    config = (tmp_path / "01-trusted-proxies.conf").read_text()
    assert "real_ip_header X-Forwarded-For;" in config
    assert "real_ip_recursive on;" in config
    assert "set_real_ip_from 172.18.0.1/32;" in config
    assert "set_real_ip_from 2001:db8::1/128;" in config


@pytest.mark.parametrize(
    "networks",
    ["0.0.0.0/0", "0.0.0.1/0", "::/0", "0:0:0:0:0:0:0:0/0",
     "0.0.0.0/00", "::/00", "127.0.0.1; include /tmp/bad;"],
)
def test_unsafe_proxy_configuration_fails_without_replacing_previous_config(tmp_path, networks):
    path = tmp_path / "01-trusted-proxies.conf"
    path.write_text("previous config")
    assert _render(tmp_path, networks).returncode != 0
    assert path.read_text() == "previous config"
