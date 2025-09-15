#!/bin/bash
set -e
CERT_DIR="nginx/certs"
DOMAIN="cicop.local"
echo "Generating self-signed SSL certificates for $DOMAIN..."
mkdir -p "$CERT_DIR"
openssl genrsa -out "$CERT_DIR/cicop.key" 2048
openssl req -new -key "$CERT_DIR/cicop.key" -out "$CERT_DIR/cicop.csr" -subj "/C=US/ST=State/L=City/O=CICOP/OU=Development/CN=$DOMAIN"
openssl x509 -req -days 365 -in "$CERT_DIR/cicop.csr" -signkey "$CERT_DIR/cicop.key" -out "$CERT_DIR/cicop.crt"
chmod 600 "$CERT_DIR/cicop.key"
chmod 644 "$CERT_DIR/cicop.crt"
rm "$CERT_DIR/cicop.csr"
echo "Certificates in $CERT_DIR (dev only). Add '127.0.0.1 $DOMAIN' to your hosts file if needed."
