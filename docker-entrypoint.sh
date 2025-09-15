#!/bin/bash
# docker-entrypoint.sh
set -e

wait_for_service() {
    local host=$1
    local port=$2
    local service=$3
    echo "Waiting for $service to be ready at ${host}:${port}..."
    for i in {1..60}; do
        if curl -sSf "http://${host}:${port}" >/dev/null 2>&1 || nc -z "$host" "$port" >/dev/null 2>&1; then
            echo "  $service is ready!"
            return 0
        fi
        echo "  $service is unavailable - sleeping"
        sleep 2
    done
    echo "Timed out waiting for $service" >&2
    return 1
}

if [ "$WAIT_FOR_DB" = "true" ]; then
    wait_for_service "${DB_HOST:-localhost}" "${DB_PORT:-5432}" "Database"
fi

if [ "$WAIT_FOR_REDIS" = "true" ]; then
    wait_for_service "${REDIS_HOST:-localhost}" "${REDIS_PORT:-6379}" "Redis"
fi

if [ "$INIT_DB" = "true" ]; then
    echo "Initializing database..."
    python scripts/setup_database.py || true
fi

if [ "$GENERATE_DATA" = "true" ] && [ ! -f "data/customer_insights.db" ]; then
    echo "Generating sample data..."
    python scripts/generate_standalone_data.py || true
fi

case "$1" in
    api)
        echo "Starting API server..."
        exec uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
        ;;
    dashboard)
        echo "Starting Streamlit dashboard..."
        exec streamlit run src/dashboard/app.py --server.port=8501 --server.address=0.0.0.0
        ;;
    worker)
        echo "Starting Celery worker..."
        exec celery -A src.workers.celery_app worker --loglevel=info
        ;;
    scheduler)
        echo "Starting Celery scheduler..."
        exec celery -A src.workers.celery_app beat --loglevel=info
        ;;
    full)
        echo "Starting full system (API + Dashboard)..."
        uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &
        API_PID=$!
        exec streamlit run src/dashboard/app.py --server.port=8501 --server.address=0.0.0.0
        ;;
    *)
        exec "$@"
        ;;
esac
