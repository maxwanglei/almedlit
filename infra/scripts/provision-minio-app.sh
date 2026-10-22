#!/bin/sh
set -eu

: "${MINIO_ROOT_USER:?}"
: "${MINIO_ROOT_PASSWORD:?}"
: "${AL_MEDLIT_STORAGE_ACCESS_KEY:?}"
: "${AL_MEDLIT_STORAGE_SECRET_KEY:?}"
: "${AL_MEDLIT_STORAGE_BUCKET:=al-medlit}"
if [ "$AL_MEDLIT_STORAGE_ACCESS_KEY" = "$MINIO_ROOT_USER" ]; then
    echo "The application storage account must differ from the MinIO root account" >&2
    exit 1
fi
if [ "$AL_MEDLIT_STORAGE_SECRET_KEY" = "$MINIO_ROOT_PASSWORD" ]; then
    echo "The application storage secret must differ from the MinIO root secret" >&2
    exit 1
fi

mkdir -p /root/.mc/certs/CAs
cp /certs/CAs/ca.crt /root/.mc/certs/CAs/ca.crt
mc alias set bootstrap https://minio:9000 "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD" >/dev/null
mc mb --ignore-existing "bootstrap/$AL_MEDLIT_STORAGE_BUCKET" >/dev/null
mc admin user add bootstrap "$AL_MEDLIT_STORAGE_ACCESS_KEY" "$AL_MEDLIT_STORAGE_SECRET_KEY" >/dev/null
cat > /tmp/al-medlit-policy.json <<EOF
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":["s3:GetBucketLocation","s3:ListBucket","s3:ListBucketMultipartUploads"],"Resource":["arn:aws:s3:::$AL_MEDLIT_STORAGE_BUCKET"]},{"Effect":"Allow","Action":["s3:GetObject","s3:PutObject","s3:DeleteObject","s3:AbortMultipartUpload","s3:ListMultipartUploadParts"],"Resource":["arn:aws:s3:::$AL_MEDLIT_STORAGE_BUCKET/*"]}]}
EOF
mc admin policy create bootstrap al-medlit-app /tmp/al-medlit-policy.json >/dev/null
mc admin policy attach bootstrap al-medlit-app --user "$AL_MEDLIT_STORAGE_ACCESS_KEY" >/dev/null
