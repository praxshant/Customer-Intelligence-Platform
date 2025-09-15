#!/bin/bash
set -e

echo "Deploying CICOP to Production..."

required_vars=("JWT_SECRET_KEY" "POSTGRES_PASSWORD" "REDIS_PASSWORD")
for var in "${required_vars[@]}"; do
  if [ -z "${!var}" ]; then
    echo "Missing required env var: $var"
    exit 1
  fi
done

if [ ! -f "nginx/certs/cicop.crt" ]; then
  bash scripts/generate_ssl_certs.sh
fi

docker compose -f docker-compose.prod.yml build
docker compose -f docker-compose.prod.yml up -d

echo "Waiting for services..."
sleep 15

echo "CICOP deployed. Access: https://localhost"
