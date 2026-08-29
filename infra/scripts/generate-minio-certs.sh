#!/bin/sh
set -eu

cert_dir=${1:?certificate directory is required}
ca_cert="$cert_dir/CAs/ca.crt"
server_cert="$cert_dir/public.crt"
server_key="$cert_dir/private.key"
generated_marker="$cert_dir/.al-medlit-generated"

# Keep a valid generated identity across container restarts. Regenerate the
# local CA and server certificate when less than 30 days of validity remain.
if [ -f "$ca_cert" ] && [ -f "$server_cert" ] && [ -f "$server_key" ]; then
    if openssl x509 -checkend 2592000 -noout -in "$ca_cert" >/dev/null 2>&1 \
        && openssl x509 -checkend 2592000 -noout -in "$server_cert" >/dev/null 2>&1 \
        && openssl verify -CAfile "$ca_cert" "$server_cert" >/dev/null 2>&1; then
        exit 0
    fi
fi

# Never replace operator-managed certificates. Only identities marked as
# locally generated may be rotated automatically by this helper.
if [ ! -f "$generated_marker" ] \
    && { [ -e "$ca_cert" ] || [ -e "$server_cert" ] || [ -e "$server_key" ]; }; then
    echo "Existing MinIO certificates are invalid or expire within 30 days; rotate them manually" >&2
    exit 1
fi

temporary_dir=$(mktemp -d)
trap 'rm -rf -- "$temporary_dir"' EXIT HUP INT TERM

cat >"$temporary_dir/ca.conf" <<'EOF'
[req]
distinguished_name = subject
x509_extensions = extensions
prompt = no

[subject]
CN = AL-MedLit local MinIO CA

[extensions]
basicConstraints = critical, CA:true
keyUsage = critical, keyCertSign, cRLSign
subjectKeyIdentifier = hash
authorityKeyIdentifier = keyid:always,issuer
EOF

cat >"$temporary_dir/server.conf" <<'EOF'
[req]
distinguished_name = subject
req_extensions = extensions
prompt = no

[subject]
CN = minio

[extensions]
basicConstraints = critical, CA:false
keyUsage = critical, digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @names

[names]
DNS.1 = minio
DNS.2 = localhost
IP.1 = 127.0.0.1
EOF

openssl req -x509 -newkey rsa:3072 -nodes \
    -keyout "$temporary_dir/ca.key" \
    -out "$temporary_dir/ca.crt" \
    -days 825 \
    -sha256 \
    -config "$temporary_dir/ca.conf" >/dev/null 2>&1
openssl req -new -newkey rsa:3072 -nodes \
    -keyout "$temporary_dir/private.key" \
    -out "$temporary_dir/server.csr" \
    -config "$temporary_dir/server.conf" >/dev/null 2>&1
openssl x509 -req \
    -in "$temporary_dir/server.csr" \
    -CA "$temporary_dir/ca.crt" \
    -CAkey "$temporary_dir/ca.key" \
    -CAcreateserial \
    -out "$temporary_dir/server.crt" \
    -days 397 \
    -sha256 \
    -extensions extensions \
    -extfile "$temporary_dir/server.conf" >/dev/null 2>&1

mkdir -p "$cert_dir/CAs"
: >"$generated_marker"
cp "$temporary_dir/ca.crt" "$ca_cert"
cat "$temporary_dir/server.crt" "$temporary_dir/ca.crt" >"$server_cert"
cp "$temporary_dir/private.key" "$server_key"
chmod 644 "$ca_cert" "$server_cert"
chmod 600 "$server_key" "$generated_marker"
