FROM golang:1.27.3-alpine@sha256:d200d2dd76b03a99b52651fd1325aff7dcd35422c35c01bac56571b0507f3deb AS builder

ENV CGO_ENABLED=0 GOTOOLCHAIN=local

FROM builder AS minio-build
RUN go install -trimpath \
    -ldflags "-s -w -X github.com/minio/minio/cmd.Version=2025-09-07T16:13:09Z -X github.com/minio/minio/cmd.ReleaseTag=RELEASE.2025-09-07T16-13-09Z" \
    github.com/minio/minio@v0.0.0-20250907161309-07c3a429bfed

FROM builder AS mc-build
RUN go install -trimpath \
    -ldflags "-s -w -X github.com/minio/mc/cmd.Version=2025-08-13T08:35:41Z -X github.com/minio/mc/cmd.ReleaseTag=RELEASE.2025-08-13T08-35-41Z" \
    github.com/minio/mc@v0.0.0-20250813083541-7394ce0dd2a8

FROM alpine:3.22@sha256:d406446bc05b82480756ce8f1287964d28768c23955fa9480d2a74a2eae7a654 AS runtime
RUN apk add --no-cache ca-certificates

FROM runtime AS minio
RUN apk add --no-cache curl \
    && addgroup -g 10001 minio \
    && adduser -D -u 10001 -G minio minio \
    && mkdir -p /data \
    && chown minio:minio /data
COPY --from=minio-build /go/bin/minio /usr/local/bin/minio
USER 10001:10001
ENTRYPOINT ["/usr/local/bin/minio"]

FROM runtime AS mc
COPY --from=mc-build /go/bin/mc /usr/local/bin/mc
ENTRYPOINT ["/usr/local/bin/mc"]
