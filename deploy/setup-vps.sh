#!/usr/bin/env bash
# ============================================================================
# CBAS — one-time VPS bootstrap.
# Run ONCE on the server after the repo is at /var/www/cbas and .env is filled.
#   cd /var/www/cbas && sudo bash deploy/setup-vps.sh
# Re-deploys after this are handled by the GitHub Actions workflow.
# ============================================================================
set -euo pipefail

APP_DIR=/var/www/cbas
cd "$APP_DIR"

echo "==> System packages (Node 20, Python, Nginx, Supervisor, Postgres client)"
sudo apt-get update -y
sudo apt-get install -y curl git nginx supervisor python3 python3-venv python3-pip
if ! command -v node >/dev/null || [ "$(node -v | cut -d. -f1 | tr -d v)" -lt 20 ]; then
  curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
  sudo apt-get install -y nodejs
fi

echo "==> Log directory"
sudo mkdir -p /var/log/cbas
sudo chown -R www-data:www-data /var/log/cbas

echo "==> .env check"
if [ ! -f .env ]; then
  echo "!! .env is missing. Copy .env.example to .env and fill it in first."; exit 1
fi
grep -q 'NEXT_PUBLIC_BASE_PATH="/cbas"' .env || echo '!! Reminder: set NEXT_PUBLIC_BASE_PATH="/cbas" and NEXTAUTH_URL="https://digixworkspace.com/cbas" in .env'

echo "==> Web app: install + build"
npm ci
npx prisma generate
npx prisma db push
npm run build

echo "==> AI engine: venv + deps + models"
cd "$APP_DIR/ai-engine"
python3 -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt
[ -f artifacts/lead_scoring.joblib ] || ./.venv/bin/python -m scripts.train_all
cd "$APP_DIR"

echo "==> Ownership"
sudo chown -R www-data:www-data "$APP_DIR"

echo "==> Supervisor"
sudo cp deploy/supervisor/cbas.conf /etc/supervisor/conf.d/cbas.conf
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl restart cbas-web cbas-ai || sudo supervisorctl start cbas-web cbas-ai

echo ""
echo "==> DONE. Now add deploy/nginx/cbas.conf's location blocks into your"
echo "    digixworkspace.com HTTPS server block, then: sudo nginx -t && sudo systemctl reload nginx"
echo "    Visit: https://digixworkspace.com/cbas"
