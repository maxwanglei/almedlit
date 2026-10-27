#!/bin/sh
set -eu

: "${MINIO_ROOT_USER:?}"
: "${MINIO_ROOT_PASSWORD:?}"
: "${AL_MEDLIT_STORAGE_ACCESS_KEY:?}"
: "${AL_MEDLIT_STORAGE_SECRET_KEY:?}"
: "${AL_MEDLIT_STORAGE_BUCKET:=al-medlit}"
validate_secret() {
    secret_name=$1
    secret_value=$2
    distinct=$(printf '%s' "$secret_value" | fold -w 1 | sort -u | wc -l)
    if [ "${#secret_value}" -lt 16 ] || [ "$distinct" -lt 8 ]; then
        printf '%s must contain at least 16 bytes and 8 distinct characters\n' "$secret_name" >&2
        exit 1
    fi
}
validate_secret MINIO_ROOT_PASSWORD "$MINIO_ROOT_PASSWORD"
validate_secret AL_MEDLIT_STORAGE_SECRET_KEY "$AL_MEDLIT_STORAGE_SECRET_KEY"
if [ "$AL_MEDLIT_STORAGE_ACCESS_KEY" = "$MINIO_ROOT_USER" ]; then
    echo "The application storage account must differ from the MinIO root account" >&2
    exit 1
fi
if [ "$AL_MEDLIT_STORAGE_SECRET_KEY" = "$MINIO_ROOT_PASSWORD" ]; then
    echo "The application storage secret must differ from the MinIO root secret" >&2
    exit 1
fi

export MC_CONFIG_DIR="${MC_CONFIG_DIR:-/root/.mc}"
mkdir -p "$MC_CONFIG_DIR/certs/CAs"
cp "${MINIO_CA_CERT_PATH:-/certs/ca.crt}" "$MC_CONFIG_DIR/certs/CAs/ca.crt"
mc alias set bootstrap https://minio:9000 "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD" >/dev/null
mc mb --ignore-existing "bootstrap/$AL_MEDLIT_STORAGE_BUCKET" >/dev/null
mc admin user add bootstrap "$AL_MEDLIT_STORAGE_ACCESS_KEY" "$AL_MEDLIT_STORAGE_SECRET_KEY" >/dev/null
cat > /tmp/al-medlit-policy.json <<EOF
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":["s3:GetBucketLocation","s3:ListBucket","s3:ListBucketMultipartUploads"],"Resource":["arn:aws:s3:::$AL_MEDLIT_STORAGE_BUCKET"]},{"Effect":"Allow","Action":["s3:GetObject","s3:PutObject","s3:DeleteObject","s3:AbortMultipartUpload","s3:ListMultipartUploadParts"],"Resource":["arn:aws:s3:::$AL_MEDLIT_STORAGE_BUCKET/*"]}]}
EOF
mc admin policy create bootstrap al-medlit-app /tmp/al-medlit-policy.json >/dev/null
if result=$(mc --json admin policy attach bootstrap al-medlit-app \
    --user "$AL_MEDLIT_STORAGE_ACCESS_KEY" 2>&1); then
    exit 0
else
    status=$?
    case "$result" in
        *XMinioPolicyAlreadyAttached*|*XMinioAdminPolicyChangeAlreadyApplied*) exit 0 ;;
        *) printf '%s\n' "$result" >&2; exit "$status" ;;
    esac
fi
