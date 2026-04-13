# Enterprise single-node deployment image for Bifrost.
#
# Builds the UI and Go gateway from the current repository checkout, then
# packages a runtime image that validates deployment environment variables
# before handing off to the standard Bifrost entrypoint.

FROM node:25-alpine3.23 AS ui-builder
WORKDIR /app

COPY ui/package*.json ./
RUN npm ci

COPY ui/ ./
RUN npx next build
RUN node scripts/fix-paths.js

FROM golang:1.26.1-alpine3.23 AS builder
WORKDIR /build

RUN apk add --no-cache gcc musl-dev sqlite-dev binutils binutils-gold

ENV CGO_ENABLED=1 GOOS=linux

COPY core/ ./core/
COPY framework/ ./framework/
COPY plugins/ ./plugins/
COPY transports/ ./transports/

RUN go work init && \
    go work use ./core && \
    go work use ./framework && \
    go work use ./plugins/governance && \
    go work use ./plugins/jsonparser && \
    go work use ./plugins/litellmcompat && \
    go work use ./plugins/logging && \
    go work use ./plugins/maxim && \
    go work use ./plugins/mocker && \
    go work use ./plugins/otel && \
    go work use ./plugins/semanticcache && \
    go work use ./plugins/telemetry && \
    go work use ./transports

RUN cd /build/transports && go mod download

COPY --from=ui-builder /app/out ./transports/bifrost-http/ui

ARG VERSION=unknown
RUN cd /build/transports && \
    go build \
      -ldflags="-w -s -X main.Version=v${VERSION} -extldflags '-static'" \
      -a -trimpath \
      -tags "sqlite_static" \
      -o /app/main \
      ./bifrost-http

RUN test -f /app/main || (echo "Build failed" && exit 1)

FROM alpine:3.23.3
WORKDIR /app

RUN apk add --no-cache musl libgcc ca-certificates wget

COPY --from=builder /app/main /app/main
COPY --from=builder /build/transports/docker-entrypoint.sh /app/docker-entrypoint.sh
COPY examples/dockers/enterprise-single-node/bootstrap.sh /app/bootstrap.sh
COPY examples/dockers/enterprise-single-node/config.tmpl.json /app/config.tmpl.json

ARG ARG_APP_PORT=8080
ARG ARG_APP_HOST=0.0.0.0
ARG ARG_LOG_LEVEL=info
ARG ARG_LOG_STYLE=json
ARG ARG_APP_DIR=/app/data

ENV APP_PORT=$ARG_APP_PORT \
    APP_HOST=$ARG_APP_HOST \
    LOG_LEVEL=$ARG_LOG_LEVEL \
    LOG_STYLE=$ARG_LOG_STYLE \
    APP_DIR=$ARG_APP_DIR \
    CONFIG_TEMPLATE=/app/config.tmpl.json \
    GOGC="" \
    GOMEMLIMIT=""

RUN mkdir -p "$APP_DIR/logs" && \
    adduser -D -s /bin/sh appuser && \
    chown -R appuser:appuser /app && \
    chmod +x /app/bootstrap.sh /app/docker-entrypoint.sh

USER appuser

VOLUME ["/app/data"]
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=5 \
  CMD wget --no-verbose --tries=1 -O /dev/null "http://127.0.0.1:${APP_PORT}/health" || exit 1

ENTRYPOINT ["/app/bootstrap.sh"]
CMD ["/app/main"]
