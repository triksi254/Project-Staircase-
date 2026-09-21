"""Lead scoring package.

Behavioural feature extraction for the lead-scoring engine described in the
research proposal: transforms the cleaned counsellor assessment records into a
feature matrix (passport status, funding clarity, English test, destination,
course/intake readiness, study gap, previous applications) plus the target
label (Cold/Good/Excellent -> 0/1/2).

The feature, rubric and v1 persona stages need no ML stack (numpy / pandas /
scikit-learn) and stay testable on their own; ``leads.features`` does import the
scraper's Pydantic ``models.schemas`` for ``QualificationLevel``. The ML
training step lives in ``leads.train_ml`` (scikit-learn / pandas) and is kept
separate. Live scoring configuration is ``leads.hybrid`` + ``artifacts/config.json``
(``python -m leads.live_config``).
"""