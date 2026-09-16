"""Lead scoring package.

Behavioural feature extraction for the lead-scoring engine described in the
research proposal: transforms the cleaned counsellor assessment records into a
feature matrix (passport status, funding clarity, English test, destination,
course/intake readiness, study gap, previous applications) plus the target
label (Cold/Good/Excellent -> 0/1/2).

The module is deliberately stdlib-only so it runs anywhere and stays testable;
the ML training step lives in ``leads.train_ml`` (scikit-learn / pandas) and is
kept separate so the feature, rubric and persona stages stay dependency-free.
"""