#!/usr/bin/env bash
# Deploy on a fresh Ubuntu 22.04 EC2 (t3.medium recommended).
# Prerequisite: clone the repo and `cd` into it before running.
set -euo pipefail

if [ ! -f .env ]; then
  echo "ERROR: .env not found. Copy .env.example to .env and fill in API keys."
  exit 1
fi

# 1. Install Docker + compose plugin if missing.
if ! command -v docker >/dev/null 2>&1; then
  echo "[deploy] Installing Docker..."
  sudo apt-get update
  sudo apt-get install -y ca-certificates curl gnupg
  sudo install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  sudo chmod a+r /etc/apt/keyrings/docker.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
    https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
  sudo apt-get update
  sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  sudo usermod -aG docker "$USER"
fi

# 2. Build + start.
echo "[deploy] docker compose build"
sudo docker compose build

echo "[deploy] docker compose up -d"
sudo docker compose up -d

# 3. Build FAISS index inside the api container (one-time).
echo "[deploy] Building FAISS index..."
sudo docker compose exec -T api python -m ingestion.build_index || true

# 4. Verify.
echo "[deploy] Health:"
curl -sS http://localhost:8000/health || true
echo
echo "[deploy] Ready:"
curl -sS http://localhost:8000/ready || true
echo
echo "[deploy] UI: http://<ec2-public-ip>:8501"
echo "[deploy] API: http://<ec2-public-ip>:8000/docs"
