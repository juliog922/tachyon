#!/usr/bin/env bash
# Writes the certificates of the TLS tests: an RSA root, a P-256 intermediate under it, a P-256 server certificate
# for hub.test and *.cdn.test under that, an RSA one for rsa.test under the root, and the two servers' keys. Valid
# for 100 years from the day they are made; the tests check them at a fixed time in 2027. Needs OpenSSL 3.
set -euo pipefail
out=${1:-crates/tachyon-cli/tests/fixtures/tls}
mkdir -p "$out"
cd "$out"
t=$(mktemp -d)
trap 'rm -rf "$t"' EXIT
openssl genrsa -out "$t/root.key" 2048 2>/dev/null
openssl req -x509 -new -key "$t/root.key" -subj "/CN=Tachyon Test Root" -days 36500 -sha256 \
    -addext basicConstraints=critical,CA:true -addext keyUsage=critical,keyCertSign,cRLSign -out root.pem
ca() { # name key-cmd subject issuer-cert issuer-key extensions
    eval "$2"
    openssl req -new -key "$1.key" -subj "$3" -out "$t/$1.csr"
    printf '%s\n' "$6" > "$t/$1.ext"
    openssl x509 -req -in "$t/$1.csr" -CA "$4" -CAkey "$5" -set_serial 0x$RANDOM -days 36500 -sha256 \
        -extfile "$t/$1.ext" -out "$1.pem" 2>/dev/null
}
ca inter "openssl ecparam -name prime256v1 -genkey -noout -out inter.key" "/CN=Tachyon Test Intermediate" root.pem "$t/root.key" \
    "basicConstraints=critical,CA:true
keyUsage=critical,keyCertSign"
ca hub "openssl ecparam -name prime256v1 -genkey -noout -out hub.key" "/CN=hub.test" inter.pem inter.key \
    "basicConstraints=critical,CA:false
keyUsage=critical,digitalSignature
extendedKeyUsage=serverAuth
subjectAltName=DNS:hub.test,DNS:*.cdn.test"
ca rsa "openssl genrsa -out rsa.key 2048 2>/dev/null" "/CN=rsa.test" root.pem "$t/root.key" \
    "basicConstraints=critical,CA:false
keyUsage=critical,digitalSignature,keyEncipherment
extendedKeyUsage=serverAuth
subjectAltName=DNS:rsa.test"
rm -f inter.key
