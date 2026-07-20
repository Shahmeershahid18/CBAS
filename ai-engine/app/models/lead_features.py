"""
Canonical feature definition for the Lead Scoring model.

Both the training pipeline and the inference service import this module so they
can never disagree about which columns the model expects. The API accepts the
clean `internal` keys; training also knows how to rename the raw Kaggle
X-Education column names onto those keys.
"""

# Internal feature key -> raw column name in the Kaggle "Lead Scoring.csv".
# Dataset: amritachatterjee09/lead-scoring-dataset (X Education case study).
KAGGLE_COLUMN_MAP = {
    "total_visits": "TotalVisits",
    "time_on_site": "Total Time Spent on Website",
    "page_views_per_visit": "Page Views Per Visit",
    "lead_origin": "Lead Origin",
    "lead_source": "Lead Source",
    "do_not_email": "Do Not Email",
    "last_activity": "Last Activity",
    "occupation": "What is your current occupation",
    "specialization": "Specialization",
}

NUMERIC_FEATURES = [
    "total_visits",
    "time_on_site",
    "page_views_per_visit",
]

CATEGORICAL_FEATURES = [
    "lead_origin",
    "lead_source",
    "do_not_email",
    "last_activity",
    "occupation",
    "specialization",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET = "converted"          # internal name
KAGGLE_TARGET = "Converted"   # raw column in the CSV
