# CBAS AI Module — Viva Defence Notes

*Why the current results are strong, correct, and defensible — and how to answer
the questions an examiner is most likely to ask.*

---

## 1. The results at a glance

| Model | Algorithm | Test-set result | Dataset |
|-------|-----------|-----------------|---------|
| Lead Scoring | XGBoost | Accuracy 81.3%, ROC-AUC 0.879, Recall 0.82 | X-Education (9,240 leads) |
| Churn (benchmark) | XGBoost | Accuracy 75.7%, ROC-AUC 0.840, Recall 0.78 | Telco (7,043 customers) |
| Churn (CRM-native) | XGBoost | Accuracy 93.4%, ROC-AUC 0.972 | CBAS engagement data (453) |
| Sentiment | TF-IDF + LogReg | Accuracy 84.4%, macro-F1 0.721 | Product reviews (137k) |
| Recommendations | Truncated-SVD CF | Precision@5 0.149 (benchmark) | Online Retail (268k tx) |

Every number is measured on a **held-out test set** the model never saw during
training (stratified 80/20 split).

---

## 2. Why these are the *right* numbers, not disappointing ones

**The scores match the published ceilings for these datasets.** These are
standard academic benchmarks with known limits:

- **Telco churn** is famously hard — published studies land around **78–82%
  accuracy / 0.83–0.85 ROC-AUC**. Our **0.840 ROC-AUC** sits squarely in that
  band. Nobody honestly reports 95% on Telco.
- **X-Education lead scoring** typically reaches **0.85–0.90 ROC-AUC**. Our
  **0.879** is right there.
- **3-class review sentiment** with classical features typically gives
  **~80–88% accuracy**. Our **84.4%** is solid, with the neutral class being the
  known bottleneck (genuinely ambiguous text).

**The key message for the panel:** *accuracy is bounded by the information in the
data, not by model effort.* We proved this empirically — a `RandomizedSearchCV`
tuning pass moved lead/churn by **less than 1%**. Reaching "95%" on these
datasets would require overfitting or data leakage, which is exactly what a good
model-evaluation section should avoid.

---

## 3. Anticipated viva questions & answers

**Q: Why isn't the accuracy 90–95%?**
A: Because these public datasets have a natural performance ceiling — human
behaviour (whether a lead converts, whether a customer churns) is only partly
predictable from the recorded features. Published work on Telco and X-Education
reports the same range we achieved. Pushing higher would mean overfitting the
test set, which would *look* better but *generalise* worse. We deliberately
report honest, generalisable numbers.

**Q: Churn accuracy is only 75.7% — isn't that weak?**
A: Raw accuracy is the wrong headline for churn. Only ~26% of Telco customers
churn, so a lazy model that predicts "nobody churns" scores 74% accuracy while
being useless. We instead optimise to **catch churners**: our **recall is 0.78**
and **ROC-AUC 0.84**, meaning we correctly flag ~78% of customers who actually
leave. For a retention use-case, missing an at-risk customer costs far more than
a false alarm, so recall and ROC-AUC are the metrics that matter — and they are
strong.

**Q: Your CRM churn model gets 93% — why is it so much higher than 75%?**
A: Two different things. The 75% Telco model is validated on real, noisy,
external telecom data. The 93% CRM model is trained on our *own* system's
engagement signals, where the relationship between disengagement and lapse is
cleaner. We present both deliberately: the benchmark proves the **technique** on
hard real data; the CRM model proves the **integration** with our actual
database. We are transparent that the CRM data is partly generated, so its higher
score reflects cleaner structure, not a "better" algorithm.

**Q: How do I know you didn't overfit or leak?**
A: Three safeguards. (1) A stratified train/test split — all metrics are on data
the model never saw. (2) For the CRM churn model, the feature that *defines* the
label (recency of last activity) is **excluded** from the inputs, so the model
must predict from independent engagement signals — its top features are note
sentiment (0.31) and activity count (0.24), not the label itself. (3) A
Logistic-Regression / Random-Forest baseline is trained alongside each improved
model, so the gain from the stronger model is measured, not assumed.

**Q: Could lead-scoring accuracy be higher? Many notebooks report ~92% on this dataset.**
A: Yes — and that ~92% is a **data-leakage trap** we deliberately avoided. We
tested it directly on the X-Education data:

| Feature set | Accuracy | ROC-AUC |
|-------------|----------|---------|
| Our features (behaviour, source, occupation…) | 82.1% | 0.878 |
| + every *legitimate* extra column | 81.8% | 0.882 |
| + *leaky* columns (Tags, Lead Quality) | **91.3%** | 0.974 |

The jump to 91% comes entirely from the `Tags` and `Lead Quality` columns, which
are filled in by **sales staff after they have already worked the lead**. For
example the tag "Closed by Horizzon" has a 99% conversion rate and "Will revert
after reading the email" 97% — these describe the *outcome*, so they are not
available when scoring a genuinely new lead. Using them is leakage: the model
"predicts" conversion from a proxy of conversion. Adding every *honest* feature,
by contrast, changed accuracy by essentially nothing (82.1% → 81.8%), which shows
our model is already at the legitimate performance ceiling for this data. **We
report the deployable 82%, not the leaky 92%** — and can prove the difference on
demand.

**Q: Why XGBoost over the baseline?**
A: We always train an interpretable baseline first (Logistic Regression for lead
scoring, Random Forest for churn) and only keep XGBoost when it wins on ROC-AUC
on the held-out set. This shows the improvement is empirical, and the baseline
gives an interpretable reference (feature weights) for explaining *why* a lead or
customer scored the way it did.

**Q: Recommendation Precision@5 is only ~0.15 — is that bad?**
A: No — for a recommender over **3,684 products**, a random guess scores about
**0.001**, so 0.149 is ~150× better than chance. Precision@K is inherently low
with large catalogues because each user has only bought a handful of items.
Recall@K and the qualitative results (coherent related products) confirm it
works.

**Q: Is the sentiment neutral class a problem?**
A: It's the honest hard part of 3-class sentiment — neutral text is ambiguous
even for humans, and there are far fewer neutral examples. Positive and negative
are classified at **0.92 and 0.82 F1**. We report per-class scores rather than
hiding behind the headline accuracy, which demonstrates evaluation maturity. (An
optional DistilBERT transformer upgrade raises overall accuracy further — see the
report.)

**Q: You tried DistilBERT — did the transformer beat the classical model?**
A: Yes. Fine-tuned on GPU (48k reviews, 2 epochs) and evaluated on the *same
natural distribution* as the baseline, DistilBERT reaches **85.9% accuracy vs
84.4%** and **macro-F1 0.731 vs 0.721** — a genuine, like-for-like improvement.
Interestingly, even the transformer scores only ~0.44 F1 on the **neutral**
class, which proves that class is inherently ambiguous rather than a weakness of
the simpler model. We keep the lightweight TF-IDF model as the *served* default
(it needs no GPU at inference) and offer DistilBERT as the higher-accuracy
option — a deliberate deployment trade-off. This demonstrates both transfer
learning and critical, distribution-aware evaluation.

**Q: Why isn't sentiment 90%+ even with DistilBERT?**
A: Three-class review sentiment has a real ceiling driven by the neutral class —
neutral reviews are ambiguous even to human annotators, and both our models plus
the published literature plateau in the mid-to-high 80s. 85.9% with a fine-tuned
transformer is a strong, honest result; claiming 95% on 3-class sentiment would
be a red flag.

**Q: Did you train a large language model from scratch?**
A: No, and we shouldn't — that's infeasible and unnecessary for an FYP. Four
features are classic supervised ML/NLP models trained from labelled data; the
chatbot integrates a pretrained LLM (Gemini) through retrieval-augmented
generation. This is the realistic, defensible engineering choice.

---

## 4. Why the benchmark + app-native design is a strength, not a hedge

We deliberately keep two models each for churn and recommendations:

- **Benchmark model** (Kaggle data): proves the algorithm works on real,
  independent, peer-recognised data — external validity.
- **App-native model** (CBAS data): proves the feature engineering and
  end-to-end integration with the product's real schema — internal validity.

Being explicit that the app-native behaviour is generated (a cold-start
mitigation) and will become fully data-driven as usage accumulates shows
engineering honesty — which examiners reward over inflated single numbers.

---

## 5. Evaluation integrity checklist (say this if asked about rigour)

- ✅ Stratified 80/20 train/test split; metrics only on unseen test data.
- ✅ Class imbalance handled (`class_weight` / `scale_pos_weight`), not ignored.
- ✅ Baseline vs improved model reported for every classifier.
- ✅ Multiple metrics per model (Accuracy, Precision, Recall, F1, ROC-AUC,
  confusion matrix) — not accuracy alone.
- ✅ Label-defining features excluded from CRM churn inputs (no leakage).
- ✅ Hyperparameter tuning attempted and its (small) effect reported honestly.
- ✅ Figures generated from the metrics (ROC curves, confusion matrices, feature
  importances) for reproducible evidence.

---

## 6. One-sentence defence

> *"Our metrics match the established performance ceilings for these benchmark
> datasets, are measured on held-out test sets with proper handling of class
> imbalance and no leakage, and are reported with full per-class detail — so they
> are honest, reproducible, and generalisable rather than inflated."*
