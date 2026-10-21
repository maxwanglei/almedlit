#!/bin/sh
set -efu

output_dir=${NGINX_ENVSUBST_OUTPUT_DIR:-/etc/nginx/conf.d}
output="$output_dir/01-trusted-proxies.conf"
temporary=$(mktemp "$output.XXXXXX")
trap 'rm -f "$temporary"' EXIT HUP INT TERM

printf '%s\n' '# Generated from AL_MEDLIT_TRUSTED_PROXY_CIDRS.' > "$temporary"
if [ -n "${AL_MEDLIT_TRUSTED_PROXY_CIDRS:-}" ]; then
    printf '%s\n' 'real_ip_header X-Forwarded-For;' 'real_ip_recursive on;' >> "$temporary"
    for network in $AL_MEDLIT_TRUSTED_PROXY_CIDRS; do
        case "$network" in
            *[!0-9a-fA-F:./]*|*/0*)
                printf '%s\n' 'Invalid trusted proxy address or unrestricted network' >&2
                exit 1
                ;;
        esac
        printf 'set_real_ip_from %s;\n' "$network" >> "$temporary"
    done
fi
mv "$temporary" "$output"
