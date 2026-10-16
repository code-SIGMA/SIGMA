#!/usr/bin/env bash
set -euo pipefail

ssh-keygen -A

mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"
if [ -f "$HOME/.ssh/authorized_keys" ]; then
    chmod 600 "$HOME/.ssh/authorized_keys"
fi

if [ -f "$HOME/.ssh/id_rsa" ]; then
    chmod 600 "$HOME/.ssh/id_rsa"
fi

if [ -f "$HOME/.ssh/id_ed25519" ]; then
    chmod 600 "$HOME/.ssh/id_ed25519"
fi

if [ -f "$HOME/.ssh/id_ed25519.pub" ]; then
    chmod 644 "$HOME/.ssh/id_ed25519.pub"
fi

if [ -f "$HOME/.ssh/id_rsa.pub" ]; then
    chmod 644 "$HOME/.ssh/id_rsa.pub"
fi

if [ -f "$HOME/.ssh/known_hosts" ]; then
    chmod 644 "$HOME/.ssh/known_hosts"
fi

chown -R $(whoami):$(whoami) "$HOME/.ssh"

exec /usr/sbin/sshd -D -e
