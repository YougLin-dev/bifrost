#!/bin/sh
set -eu

require_env() {
    var_name="$1"
    eval "var_value=\${$var_name:-}"
    if [ -z "$var_value" ]; then
        echo "error: required environment variable '$var_name' is not set" >&2
        exit 1
    fi
}

APP_DIR="${APP_DIR:-/app/data}"
CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-/app/config.tmpl.json}"
CONFIG_PATH="${APP_DIR}/config.json"

for required_var in \
    POSTGRES_USER \
    POSTGRES_PASSWORD \
    POSTGRES_DB \
    BIFROST_DB_HOST \
    BIFROST_DB_PORT \
    BIFROST_DB_SSLMODE \
    BIFROST_ADMIN_USERNAME \
    BIFROST_ADMIN_PASSWORD \
    BIFROST_ENCRYPTION_KEY
do
    require_env "$required_var"
done

if [ ! -f "$CONFIG_TEMPLATE" ]; then
    echo "error: config template not found at '$CONFIG_TEMPLATE'" >&2
    exit 1
fi

umask 077
mkdir -p "$APP_DIR" "$APP_DIR/logs"
cp "$CONFIG_TEMPLATE" "$CONFIG_PATH"
chmod 600 "$CONFIG_PATH"

exec /app/docker-entrypoint.sh "$@"
