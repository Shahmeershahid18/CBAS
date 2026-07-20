# CBAS AI Module Implementation Plan

**Project:** Cognitive Business Automation System (CBAS)
**Module:** AI Engine (analytics, prediction, sentiment, recommendation, chatbot)
**Prepared for:** FYP Report, Section 7 (AI Module Development)

---

## 1. Scope and Positioning

CBAS does not require building or training a Large Language Model from scratch. That path is not feasible within an FYP in cost, compute, or timeline, and it is not what the system needs.

The AI Engine is a set of small, well understood models. Four of the five AI features are classic Machine Learning and NLP models trained on tabular or text data. The fifth feature, the chatbot, is built by integrating an existing pretrained language model through an API, or by training intent classification in an open source framework. In both chatbot cases no language model is trained from zero.

This positioning keeps the project realistic, defensible in a viva, and fully achievable on free compute (Google Colab or Kaggle Notebooks).

---

## 2. Component to Model Mapping

| # | CBAS Feature | Problem Type | Model Approach | Trained From Scratch? |
|---|--------------|--------------|----------------|-----------------------|
| 1 | AI-Driven Lead Scoring | Binary classification | Logistic Regression baseline, then XGBoost | No, standard ML training |
| 2 | Predictive Analytics (Churn) | Binary classification | Random Forest / XGBoost | No, standard ML training |
| 3 | Sentiment Analysis | Text classification (NLP) | TF-IDF + Logistic Regression baseline, optional DistilBERT fine-tune | No, fine-tune a pretrained model |
| 4 | Personalized Recommendations | Recommender system | Collaborative filtering (matrix factorization) | No |
| 5 | Intelligent Chatbot | Conversational AI | Rasa intent model, or LLM API with retrieval | No, integrate a pretrained LLM |

---

## 3. Component Detail

### 3.1 Lead Scoring
Predicts whether a lead will convert, then ranks leads by predicted probability so the sales team works the highest value leads first.

* **Input features:** lead source, time on site, activity, occupation, page views, and similar behavioural fields.
* **Target:** `Converted` (1 or 0).
* **Baseline:** Logistic Regression for interpretability, since feature weights explain why a lead scored high.
* **Improved model:** XGBoost for higher accuracy and feature importance charts.
* **Output to system:** a 0 to 100 score per lead, written back to the CRM lead record.

### 3.2 Predictive Analytics (Churn)
Flags customers likely to stop buying so the business can act early with retention offers.

* **Input features:** tenure, contract type, monthly charges, services subscribed, payment method.
* **Target:** `Churn` (Yes or No).
* **Model:** Random Forest or XGBoost.
* **Output to system:** churn risk band (Low, Medium, High) per customer, surfaced on the dashboard.

### 3.3 Sentiment Analysis
Reads customer reviews, emails, and feedback and labels them positive, negative, or neutral, feeding a satisfaction gauge on the dashboard.

* **Baseline:** text cleaning, TF-IDF vectorization, Logistic Regression. Fast, light, easy to defend.
* **Optional upgrade:** fine-tune DistilBERT from Hugging Face on a GPU (free Colab tier is sufficient) for higher accuracy.
* **Output to system:** sentiment label plus confidence, aggregated into a satisfaction trend.

### 3.4 Personalized Recommendations
Suggests products or services per customer using purchase history.

* **Approach:** collaborative filtering with matrix factorization using the `surprise` or `implicit` library.
* **Input:** user to item interaction data (purchases, ratings, views).
* **Output to system:** top N recommended items per customer.

### 3.5 Intelligent Chatbot
Handles FAQs, order tracking, and basic support around the clock. Two paths, choose one:

**Path A, Rasa (open source):**
Train an intent classifier and entity extractor on a few hundred example utterances the team writes. This is genuine training work you can present, and it matches the tools already listed in the proposal.

**Path B, LLM API with Retrieval (recommended for a stronger demo):**
Call an existing LLM (OpenAI or Gemini) and connect it to CRM data using retrieval augmented generation (embeddings plus a vector store). The bot answers questions about real customers, leads, and orders inside CBAS. The model is used, not trained.

Recommendation: Path B gives the most impressive live demo with the least data collection effort. Path A is the better choice if the requirement is to show a model the team trained end to end.

---

## 4. Datasets (Kaggle)

All structured and text datasets below are available on Kaggle. The chatbot has no downloadable training dataset and is handled per Section 3.5.

| Component | Recommended Kaggle Dataset | Notes |
|-----------|----------------------------|-------|
| Lead Scoring | `amritachatterjee09/lead-scoring-dataset` or `lakshmikalyan/lead-scoring-x-online-education` | X Education case study, around 9,000 leads with `Converted` label |
| Churn | `blastchar/telco-customer-churn` | 7,043 customers, standard benchmark. Variant with review text: `beatafaron/telco-customer-churn-realistic-customer-feedback` |
| Sentiment | `bittlingmayer/amazonreviews` (large) or `mansithummar67/171k-product-review-with-sentiment-dataset` (lighter) | Labeled review text |
| Recommendations | Search Kaggle for "online retail" or "e-commerce user behavior" transactions | Needs user to item interactions |

**Data honesty note:** some newer Kaggle sentiment and feedback sets are synthetic (AI generated). They work for a functioning prototype and demo, but if a supervisor asks, state clearly that the data is synthetic. Real world sets (Amazon, Telco) make for a stronger defense.

---

## 5. Training Pipeline (per model)

The same repeatable pipeline applies to lead scoring, churn, and sentiment:

1. Load and clean the dataset (handle missing values, drop leakage columns).
2. Encode categorical features and vectorize text where relevant.
3. Split into train and test sets (typical 80 / 20, stratified on the target).
4. Train a simple baseline first.
5. Train the improved model and tune key hyperparameters.
6. Evaluate on the held out test set.
7. Save the trained model with `joblib` (or the Hugging Face format for DistilBERT).
8. Serve the model behind a FastAPI endpoint.

---

## 6. Evaluation Metrics

| Model Type | Primary Metrics |
|------------|-----------------|
| Lead Scoring, Churn | Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix |
| Sentiment | Accuracy, macro F1, per class precision and recall |
| Recommendations | Precision@K, Recall@K, RMSE on held out ratings |
| Chatbot (Rasa) | Intent accuracy, entity F1 on a test conversation set |

Report a confusion matrix and a feature importance chart for the classification models. These make strong figures in the report and defense.

---

## 7. Architecture and Integration

The AI Engine sits as a separate Python service so it does not couple to the main backend language.

```
Frontend (React)
      |
Backend (Node.js / Django)  --->  AI Engine (FastAPI, Python)
      |                                 |
   Database                        Trained models (joblib / HF)
                                        |
                                   LLM API (chatbot, Path B)
```

* Each model is exposed as a REST endpoint (for example `POST /score-lead`, `POST /predict-churn`, `POST /analyze-sentiment`, `GET /recommendations/{customer_id}`).
* The main backend calls these endpoints and stores results against CRM records.
* This FastAPI service is the "AI Engine" box in the proposal block diagram.

This design fits the existing stack (Python, TensorFlow, Scikit-learn already listed) and keeps model code isolated and testable.

---

## 8. Environment and Tools

* **Training:** Google Colab or Kaggle Notebooks (free GPU available for the DistilBERT option).
* **Core libraries:** pandas, scikit-learn, XGBoost, `surprise` or `implicit`, Hugging Face Transformers (optional), Rasa (Path A) or an LLM SDK (Path B).
* **Serving:** FastAPI, joblib.
* **Version control:** GitHub, per the proposal.

---

## 9. Mapping to Proposal Timeline

| Proposal Phase | AI Module Work |
|----------------|----------------|
| Month 7 (AI Module Development) | Lead scoring and churn models: data prep, training, evaluation, save |
| Month 8 (AI Module Development) | Sentiment model and recommender, then wrap all models in FastAPI endpoints |
| Month 9 (Chatbot Integration) | Build chatbot (Rasa or LLM plus retrieval), connect to CRM data |

---

## 10. Open Items to Verify

* Confirm final Kaggle dataset choice per component and check the license on each.
* Decide chatbot Path A (Rasa) or Path B (LLM plus retrieval) based on supervisor expectations.
* Confirm whether synthetic data is acceptable for the sentiment component.
* Finalize the recommender dataset (search and select a suitable transactions set).
