# CBAS Deployment — digixworkspace.com/cbas

Deploys **two** processes on your VPS, managed by **Supervisor**, behind **Nginx**:

| Process | What | Port (localhost) | Public |
|---------|------|------------------|--------|
| `cbas-web` | Next.js app (`next start`) | 3050 | Yes — at `/cbas` |
| `cbas-ai` | Python AI Engine (`uvicorn`) | 8088 | No (internal only) |

The app is served under the **sub-path** `/cbas`, so Next.js is built with
`basePath=/cbas` (via `NEXT_PUBLIC_BASE_PATH`) and NextAuth is told to use
`/cbas/api/auth`.

---

## 1. GitHub secrets (Settings → Secrets and variables → Actions)

| Secret | Value |
|--------|-------|
| `VPS_HOST` | server IP / hostname |
| `VPS_USER` | SSH user — needs **write to `/var/www`** and passwordless `sudo supervisorctl` (root satisfies both) |
| `VPS_SSH_KEY` | the **private** key whose public key is in the server's `~/.ssh/authorized_keys` |
| `VPS_PORT` | SSH port (optional, default 22) |
| `REPO_PAT` | GitHub token with `repo` read scope — **only if the repo is PRIVATE**. Leave unset for a public repo. |

The workflow now **clones the repo on the first run and pulls after that** — you no
longer clone by hand. On the very first run it will clone, then stop with a
message if `/var/www/cbas/.env` is missing; create `.env` (and run the one-time
`setup-vps.sh` for system packages + Supervisor/Nginx), then re-run the workflow.

Give the deploy user passwordless sudo for just supervisor:
```bash
echo '<VPS_USER> ALL=(ALL) NOPASSWD: /usr/bin/supervisorctl' | sudo tee /etc/sudoers.d/cbas-deploy
```

## 2. Server `.env` (in `/var/www/cbas/.env`)

Critical values for sub-path hosting:
```bash
NEXT_PUBLIC_BASE_PATH="/cbas"
NEXTAUTH_URL="https://digixworkspace.com/cbas"
NEXTAUTH_URL_INTERNAL="http://127.0.0.1:3050"
DATABASE_URL="postgresql://USER:PASS@localhost:5432/cbas?schema=public"
AI_ENGINE_URL="http://127.0.0.1:8088"
GEMINI_API_KEY="..."          # for the chatbot
# ...plus the rest from .env.example
```

## 3. One-time bootstrap (on the VPS)

```bash
sudo mkdir -p /var/www/cbas && sudo chown -R $USER /var/www/cbas
git clone <your-repo-url> /var/www/cbas
cd /var/www/cbas
cp .env.example .env && nano .env      # fill it in (see §2)
sudo bash deploy/setup-vps.sh
```

Then wire up Nginx — paste the `location` blocks from
[`deploy/nginx/cbas.conf`](nginx/cbas.conf) into your existing
`digixworkspace.com` HTTPS server block, add the `map $http_upgrade …` snippet to
`nginx.conf` if missing, then:
```bash
sudo nginx -t && sudo systemctl reload nginx
```
Visit **https://digixworkspace.com/cbas** 🎉

## 4. Ongoing deploys

Just `git push` to `main`. The [workflow](../.github/workflows/deploy.yml) SSHes
in, pulls, installs, runs `prisma db push`, rebuilds the web app, updates the AI
engine's venv, and restarts both Supervisor programs. Trigger manually anytime
from the **Actions** tab (`workflow_dispatch`).

## 5. Useful commands

```bash
sudo supervisorctl status                 # both processes
sudo supervisorctl restart cbas-web cbas-ai
tail -f /var/log/cbas/web.err.log         # Next.js logs
tail -f /var/log/cbas/ai.err.log          # AI engine logs
curl -s http://127.0.0.1:8088/health      # AI engine health
```

## 6. AI models on the server

Trained model artifacts (`*.joblib`) are git-ignored, so the bootstrap/deploy
runs `scripts.train_all` (synthetic fallback) to create working models. For the
**real** benchmark metrics and the **app-native** churn/recs/sentiment models,
after the DB is seeded run once on the server:
```bash
cd /var/www/cbas
npm run db:seed:ai && npm run ai:export
cd ai-engine
./.venv/bin/python -m training.churn_crm
./.venv/bin/python -m training.sentiment_crm
./.venv/bin/python -m training.recommend --data data/crm_orders.csv
sudo supervisorctl restart cbas-ai
```

---

## ⚠️ Sub-path caveats (important)

Hosting a Next.js + NextAuth app under `/cbas` is fussier than a root domain:
- **All links/assets** must respect `basePath` — already handled by the
  `basePath`/`assetPrefix` config. Don't hard-code `/dashboard`; use Next's
  `<Link>`/`router` which prepend the base path automatically.
- **NextAuth** must have `NEXTAUTH_URL` include `/cbas`, and the SessionProvider
  `basePath` is set from `NEXT_PUBLIC_BASE_PATH` (done in `providers.tsx`).
- If you hit auth redirect loops, double-check those two values.

### Simpler alternative — a sub-domain
A **sub-domain** (`cbas.digixworkspace.com`) avoids every sub-path gotcha: no
`basePath`, `NEXTAUTH_URL="https://cbas.digixworkspace.com"`, and Nginx becomes a
plain `server { server_name cbas.digixworkspace.com; location / { proxy_pass
http://127.0.0.1:3050; } }` with its own Certbot cert. If you can add a DNS
record, this is the recommended route — tell me and I'll adjust the configs.
