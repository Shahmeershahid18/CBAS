# CBAS AI Engine

The standalone **AI Engine** for CBAS — a Python/FastAPI service that trains and
serves the platform's machine-learning models over REST. The Next.js backend
calls these endpoints and stores the results against CRM records. This keeps the
model code isolated, testable, and off the main Node.js runtime.

```
Next.js backend  ──HTTP──▶  AI Engine (FastAPI, this service)
                                 │
                          Trained models (joblib)
```

## Modules

| Endpoint | Model | Status |
|----------|-------|--------|
| `POST /score-lead` | Lead Scoring — Logistic Regression baseline → XGBoost | ✅ Implemented |
| `POST /predict-churn` | Churn — Random Forest baseline → XGBoost | ✅ Implemented |
| `POST /analyze-sentiment` | Sentiment — TF-IDF + Logistic Regression (3-class) | ✅ Implemented |
| `GET  /recommendations/{id}` | Recommendations — matrix-factorization collaborative filtering | ✅ Implemented |
| `POST /predict-churn-crm` | Churn (CRM-native) — trained on CBAS's own engagement data | ✅ Implemented |
| `POST /analyze-sentiment-crm` | Sentiment (CRM-native) — trained on business/sales note language | ✅ Implemented |
| _(in the Next.js app)_ | Chatbot — Gemini LLM + RAG retrieval over CRM records | ✅ Implemented |

Sentiment, like churn and recommendations, has a **benchmark** model (product
reviews) and an **app-native** model (CRM notes). The app uses the CRM-native
model for activity notes, because the review model mislabels business language
("signed the contract", "renewed early"). The recommender is trained on the
app's own orders, whose catalogue is the software house's **digital services**
(web/mobile, cloud/DevOps, marketing/SEO, software/support), and uses item-based
collaborative filtering so suggestions reflect each customer's own purchases.

Churn and recommendations exist in **two flavours**: a *benchmark* model trained
on public Kaggle data (proves the technique), and an *app-native* model trained
on CBAS's own database (runs on real customers/products). See
"App-native models" below.

Each new model module here follows the same shape: a `training/<name>.py`
pipeline that saves a joblib bundle, an `app/models/<name>.py` inference
wrapper, and a route in `app/main.py`.

**Chatbot (module 5)** lives in the Next.js app rather than this Python service,
because it integrates a hosted LLM (the plan's *Path B*) instead of a locally
trained model:
- `askAssistant()` in [`src/lib/actions/chatbot.ts`](../src/lib/actions/chatbot.ts) —
  Gemini, grounded in workspace metrics + confirmable tool-actions.
- `retrieveContext()` in [`src/lib/ai/retrieval.ts`](../src/lib/ai/retrieval.ts) —
  **RAG**: embeds the workspace's leads/contacts/deals with the local MiniLM
  model ([`src/lib/ai/embeddings.ts`](../src/lib/ai/embeddings.ts), cached by
  record id+updatedAt) and injects the top-k semantically-relevant records into
  the prompt, so the bot answers about specific records, not just totals.
- Requires `GEMINI_API_KEY` in the app's `.env`.

## Results on real Kaggle data

Held-out test metrics after training on the real datasets (see "Real datasets"
below). Reproduce with `python -m scripts.train_all`.

| Model | Dataset (rows) | Best model | Headline metric |
|-------|----------------|-----------|-----------------|
| Lead Scoring | X-Education (9,240 leads) | XGBoost | ROC-AUC **0.876**, F1 0.768 |
| Churn | Telco (7,043 customers) | XGBoost | ROC-AUC **0.839**, F1 0.619 |
| Sentiment | Product reviews (171,378) | TF-IDF + LogReg | Accuracy **0.844**, macro-F1 0.721 |
| Recommendations | Online Retail (267,615 tx, 3,684 items) | Truncated-SVD CF | Precision@5 **0.149**, Recall@5 0.086 |

Full per-model breakdowns (per-class scores, confusion matrices, feature
importances) are written to `artifacts/*_metrics.json`.

## Real datasets (one-shot)

With a Kaggle API token in place (`~/.kaggle/kaggle.json` — see
`scripts/fetch_datasets.py` for how to create one):

```bash
python -m scripts.fetch_datasets   # downloads all four CSVs into data/
python -m scripts.train_all        # retrains every model on the real data
```

`train_all` uses the real CSV when present in `data/` and the synthetic fallback
otherwise, so it always runs. Per-dataset details are in the sections below.

## Setup

```bash
cd ai-engine
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env        # then set AI_ENGINE_API_KEY
```

## Train the lead-scoring model

**Option A — real Kaggle data (use this for the report/defence):**

1. Download the [X-Education Lead Scoring dataset](https://www.kaggle.com/datasets/amritachatterjee09/lead-scoring-dataset).
2. Put `Lead Scoring.csv` in `ai-engine/data/`.
3. Train:

   ```bash
   python -m training.lead_scoring
   ```

**Option B — synthetic fallback (works with no download, for smoke tests):**

```bash
python -m training.lead_scoring --synthetic
```

Both write:
- `artifacts/lead_scoring.joblib` — the full preprocessing + model pipeline
- `artifacts/lead_scoring_metrics.json` — Accuracy, Precision, Recall, F1,
  ROC-AUC, confusion matrix, and top feature importances (figures for the report)

> The synthetic set is clearly labelled as such in the metrics report. Train on
> the real Kaggle data before presenting results.

## Train the churn model

Same flow as lead scoring. Real data (recommended):
[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
→ save `Telco-Customer-Churn.csv` into `ai-engine/data/`, then:

```bash
python -m training.churn              # real Kaggle CSV if present, else synthetic
python -m training.churn --synthetic  # force synthetic
```

Writes `artifacts/churn.joblib` and `artifacts/churn_metrics.json`.

### Example churn request

```bash
curl -X POST http://127.0.0.1:8088/predict-churn \
  -H "Content-Type: application/json" \
  -d '{"tenure":1,"monthly_charges":95,"contract":"Month-to-month","internet_service":"Fiber optic","payment_method":"Electronic check","tech_support":"No"}'
# -> {"risk_score":97,"risk_band":"High", ...}
```

## How the Next.js app uses churn

- Client: `predictChurnRemote()` in [`src/lib/ai/engine.ts`](../src/lib/ai/engine.ts)
- Server action: `predictContactChurn()` in [`src/lib/actions/contacts.ts`](../src/lib/actions/contacts.ts)
- Fields written back: `churnScore` / `churnBand` / `churnReason` on `Contact`

> Data note: the CRM has no billing/subscription fields yet, so the contact's
> account age is used as `tenure` and the linked deal value as a `monthly_charges`
> proxy; the rest is imputed. Capture real billing fields for a production signal.

## Train the sentiment model

TF-IDF + Logistic Regression, three classes (positive / negative / neutral).
Real data (recommended): a labelled review set such as
[Amazon reviews](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews) or
[171k product reviews](https://www.kaggle.com/datasets/mansithummar67/171k-product-review-with-sentiment-dataset)
→ save as `ai-engine/data/reviews.csv` (any text column + a label/rating column),
then:

```bash
python -m training.sentiment              # real CSV if present, else synthetic
python -m training.sentiment --synthetic  # force synthetic
```

Writes `artifacts/sentiment.joblib` and `artifacts/sentiment_metrics.json`
(accuracy, macro-F1, per-class precision/recall, confusion matrix).

> The synthetic set uses clean templated phrases, so it scores ~1.0 — that is a
> smoke test, not a real result. Train on Amazon/product reviews for the report.

### Example sentiment request

```bash
curl -X POST http://127.0.0.1:8088/analyze-sentiment \
  -H "Content-Type: application/json" \
  -d '{"text":"Very disappointed, the app keeps crashing and support ignored me"}'
# -> {"sentiment":"negative","confidence":0.90, "scores":{...}}
```

## How the Next.js app uses sentiment

- Client: `analyzeSentimentRemote()` in [`src/lib/ai/engine.ts`](../src/lib/ai/engine.ts)
- Applied automatically when a note/activity is logged (`createActivity` in
  [`src/lib/actions/activities.ts`](../src/lib/actions/activities.ts)) — best-effort,
  so a note still saves if the engine is offline.
- Fields written back: `sentiment` / `sentimentScore` on `Activity`; a coloured
  badge shows next to the note in the lead timeline.

## Train the recommendation model

Collaborative filtering via **matrix factorization** (Truncated SVD over the
user-item matrix — the same latent-factor idea as `surprise`/`implicit`, using
only scipy/scikit-learn so it installs cleanly on Windows).

Real data (recommended): a retail transactions set such as the
[Online Retail](https://www.kaggle.com/datasets/vijayuv/onlineretail) dataset →
save as `ai-engine/data/retail.csv` (needs a user column like `CustomerID`, an
item column like `StockCode`/`Description`, optionally `Quantity`), then:

```bash
python -m training.recommend              # real CSV if present, else synthetic
python -m training.recommend --synthetic  # force synthetic
```

Writes `artifacts/recommend.joblib` and `artifacts/recommend_metrics.json`
(Precision@K and Recall@K on held-out interactions).

### Example recommendations request

```bash
curl "http://127.0.0.1:8088/recommendations/5?n=5"
# -> {"customer_id":"5","personalized":true,"recommendations":[{"name":"Wireless Charger",...}]}
```

Unknown customers get a popularity fallback (`personalized: false`).

## How the Next.js app uses recommendations

- Client: `getRecommendationsRemote()` in [`src/lib/ai/engine.ts`](../src/lib/ai/engine.ts)
- Server action: `getContactRecommendations()` in [`src/lib/actions/contacts.ts`](../src/lib/actions/contacts.ts)

> Data note: the recommender is trained on its own transactions dataset, so the
> app's contact IDs are unknown and it returns the popularity fallback. Train on
> THIS app's order history (`user_id,item_id,quantity` → `training.recommend
> --data <csv>`) so the model's user/item space matches the CRM before surfacing
> personalized picks in the UI.

## App-native models (trained on CBAS's own data)

Churn and recommendations also train directly on the app's database, so they run
on real customers and products instead of external data. Because a fresh system
has little history, an app-native dataset is generated inside the CBAS schema
first. From the **project root** (not ai-engine/):

```bash
npm run db:seed:ai     # 48 products, 400 contacts, ~1.6k orders, ~1.4k activities
npm run ai:export      # -> ai-engine/data/crm_orders.csv, crm_churn.csv
```

Then, in `ai-engine/`:

```bash
python -m training.churn_crm                       # CRM-native churn model
python -m training.recommend --data data/crm_orders.csv   # app-native recommender
```

- CRM churn is served at `POST /predict-churn-crm`; the app's `predictContactChurn`
  engineers the matching features and calls it.
- The app-native recommender overwrites `recommend.joblib` (the app needs it);
  the Kaggle benchmark is preserved as `recommend_kaggle.joblib` /
  `recommend_metrics_kaggle.json`.

> The generated behaviour is synthetic but lives in the real schema. Re-running
> the two commands above after real usage accumulates produces genuinely
> data-driven models with no code changes.

## Run the service

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8088
```

- Health + model readiness: `GET  http://127.0.0.1:8088/health`
- Model metrics:            `GET  http://127.0.0.1:8088/models/lead-scoring`
- Interactive docs:         `http://127.0.0.1:8088/docs`

### Example request

```bash
curl -X POST http://127.0.0.1:8088/score-lead \
  -H "Content-Type: application/json" \
  -H "x-ai-engine-key: <your-key>" \
  -d '{"lead_source":"Reference","occupation":"Working Professional","time_on_site":900,"total_visits":6}'
```

```json
{
  "score": 87,
  "probability": 0.8712,
  "band": "Hot",
  "model": "xgboost",
  "reason": "Hot lead: 87/100 conversion probability. ...",
  "key_factors": [ { "feature": "time_on_site", "importance": 0.23 }, ... ]
}
```

## How the Next.js app uses it

- Client: [`src/lib/ai/engine.ts`](../src/lib/ai/engine.ts)
- Server action: `scoreLead()` in [`src/lib/actions/leads.ts`](../src/lib/actions/leads.ts)
- Config: `AI_ENGINE_URL` and `AI_ENGINE_API_KEY` in the app's `.env`

All feature fields are optional in the request — the CRM sends whatever it knows
and the model's imputers fill the rest.
