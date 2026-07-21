# CBAS — Testing the AI Modules

How to run the system and test each of the five AI features end-to-end.

---

## 0. One-time prerequisites (already done on this machine)

If setting up fresh elsewhere:

```bash
# App
npm install
npx prisma db push          # create tables
npm run db:seed:demo        # demo login + CRM sample data
npm run db:seed:ai          # products, orders, engagement data (app-native models)
npm run ai:export           # -> ai-engine/data/crm_orders.csv, crm_churn.csv

# AI Engine
cd ai-engine
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m scripts.train_all             # lead, churn, sentiment, recommend (Kaggle)
python -m training.churn_crm            # app-native churn
python -m training.sentiment_crm        # app-native CRM-note sentiment
python -m training.recommend --data data/crm_orders.csv   # app-native recs (digital services)
```

---

## 1. Start the two services (two terminals)

**Terminal 1 — AI Engine (Python):**
```bash
cd ai-engine
.venv\Scripts\activate
uvicorn app.main:app --port 8088
```

**Terminal 2 — App (Next.js):**
```bash
npm run dev
```

Open <http://localhost:3000> and log in:

| Email | Password |
|-------|----------|
| `demo@digixcrm.com` | `DemoPassword2026@` |

---

## 2. Confirm the engine is healthy

Browser or curl: <http://127.0.0.1:8088/health> — every model should read `"ready": true`:

```json
{ "status":"ok", "models":{
  "lead_scoring":{"ready":true}, "churn":{"ready":true}, "churn_crm":{"ready":true},
  "sentiment":{"ready":true}, "recommendations":{"ready":true} } }
```

Interactive API docs (try any endpoint from the browser): <http://127.0.0.1:8088/docs>

On the **dashboard**, the **AI Insights** widget shows an "Engine online" badge when
the service is up (and distributions once you've scored some records).

---

## 3. Test each module

### 3.1 Lead Scoring
**UI:** Dashboard → **Leads** → click a lead → in the detail panel, the **AI Lead
Score** card → click **Score**.
**Expect:** a 0–100 score + Hot/Warm/Cold band + one-line reason; re-openable as
"Re-score".
**API:**
```bash
curl -X POST http://127.0.0.1:8088/score-lead -H "Content-Type: application/json" \
  -d '{"lead_source":"Reference","occupation":"Working Professional","time_on_site":1200,"total_visits":8}'
```
Engaged/referral → high score; sparse/do-not-email → low.

### 3.2 Churn (app-native, CRM model)
**UI:** Dashboard → **Contacts** → row menu (⋮) → **Predict Churn Risk**.
**Expect:** the **Churn Risk** column fills with a Low/Medium/High badge + score.
Try a seeded contact (email ends `@ai-seed.demo`) with many orders vs one with few.
**API (CRM model):**
```bash
curl -X POST http://127.0.0.1:8088/predict-churn-crm -H "Content-Type: application/json" \
  -d '{"tenure_days":200,"num_orders":12,"total_spend":5000,"num_activities":9,"num_negative_notes":0,"avg_sentiment":0.8,"num_categories":3,"has_open_deal":1}'
```
**API (Telco benchmark):** `POST /predict-churn` with `{"tenure":1,"monthly_charges":95,"contract":"Month-to-month",...}`

### 3.3 Sentiment Analysis
**UI:** Dashboard → **Leads** → open a lead → type a note in the activity box →
**Save**. A coloured 😊/😐/😞 **sentiment badge** appears next to the note.
Notes are analysed by the **CRM-native** model (`/analyze-sentiment-crm`), trained
on business language, so sales phrasing classifies correctly:
- "Client is thrilled, signed the contract today" → positive
- "Angry client, threatening to cancel the contract" → negative
- "Called the client, left a voicemail, awaiting reply" → neutral

**API (CRM notes — used by the app):**
```bash
curl -X POST http://127.0.0.1:8088/analyze-sentiment-crm -H "Content-Type: application/json" \
  -d '{"text":"Customer renewed early, very happy with our service"}'
```
**API (review benchmark model):** `POST /analyze-sentiment` (best on product-review text).

### 3.4 Recommendations
**UI:** Dashboard → **Contacts** → row menu (⋮) → **Recommend Products** → a dialog
lists the top products. Seeded contacts (`@ai-seed.demo`) get **personalized**
picks; contacts with no orders get a popularity fallback.
**API:**
```bash
# use a real contact id (see query below) or any id for the popularity fallback
curl "http://127.0.0.1:8088/recommendations/<contactId>?n=5"
```
Get a seeded contact id with orders:
```bash
"C:\Program Files\PostgreSQL\18\bin\psql.exe" -U cbas_admin -h localhost -d cbas -tAc \
  "SELECT \"contactId\" FROM \"Order\" GROUP BY \"contactId\" ORDER BY count(*) DESC LIMIT 1;"
```

### 3.5 Chatbot (Gemini + RAG)  — needs an API key
1. Get a free key at <https://aistudio.google.com/apikey>.
2. Put it in the project root `.env`: `GEMINI_API_KEY="your-key"` and restart `npm run dev`.
3. **UI:** the floating **chat widget** (bottom corner of the dashboard). Ask:
   - *"How many leads and deals do we have?"* (uses workspace metrics)
   - *"Which contacts are high churn risk?"* (uses retrieved records)
   - *"Log an expense of $200 for software"* → it proposes an action you **Confirm**.
Without a key the widget replies that the assistant is unavailable — that's expected.

**Tagging a lead/customer (@-mention) + privacy:**
- Type **`@`** then a name → pick a lead/contact from the dropdown → a tag chip appears.
- Ask *"How should I pitch this customer?"* / *"What's the next best action?"* → the
  assistant advises using that record's business attributes (service, value, source,
  AI score, engagement, churn risk).
- **Privacy:** the picker shows names to *you*, but only **non-identifying** attributes
  are sent to the LLM. Name, email, phone, and raw notes are **never** sent (redacted in
  `buildTaggedContext`). Tag multiple records to compare.

### 3.6 AI Insights dashboard widget
**UI:** the **Dashboard** home page. After you've scored some leads/contacts and
logged notes, the widget shows Hot/Warm/Cold, churn High/Med/Low, and note
sentiment distributions, plus the engine status badge.

---

## 4. Model metrics & figures (for the report)

```bash
# per-model metrics (accuracy, F1, ROC-AUC, confusion matrix, importances)
ai-engine/artifacts/*_metrics.json

# regenerate report figures
cd ai-engine && python -m scripts.make_figures   # -> artifacts/figures/*.png
```
Model info endpoints: `/models/lead-scoring`, `/models/churn`, `/models/churn-crm`,
`/models/sentiment`, `/models/recommendations`.

---

## 5. Retraining (after real data accumulates)

```bash
# Kaggle/benchmark models
cd ai-engine && python -m scripts.train_all

# App-native models (from the live database)
npm run ai:export                                       # export CRM CSVs
cd ai-engine
python -m training.churn_crm
python -m training.recommend --data data/crm_orders.csv
# restart uvicorn to load the new artifacts
```

---

## 6. Troubleshooting

| Symptom | Fix |
|---------|-----|
| Widget says "Engine offline" / actions fail | Start the AI Engine (step 1). Check <http://127.0.0.1:8088/health>. |
| `503 ... No trained model` | Train that model (section 0) then restart uvicorn. |
| Chatbot "unavailable" | Set `GEMINI_API_KEY` in `.env`, restart `npm run dev`. |
| Recommendations always "popularity fallback" | That contact has no orders; use a `@ai-seed.demo` contact, or re-run `npm run db:seed:ai` + `npm run ai:export` + retrain. |
| Port 8088 in use | `uvicorn app.main:app --port 8090` and set `AI_ENGINE_URL` in `.env`. |
