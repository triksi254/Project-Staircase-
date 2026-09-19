"""H1 evaluation: retrieval quality against hand-labelled gold sets.

``retrieval_eval`` scores the existing ``chatbot.retriever.Retriever`` on
rank-1 accuracy, MRR, recall@k, abstention behaviour and a threshold sweep.
``adaptation_loop`` simulates a cold-start corpus reduction and measures how
much of the resulting retrieval loss a targeted top-up recovers.

Gold ``gold_id`` values are the **raw JSON index** of the FAQ entry, i.e.
``FaqEntry.index`` from ``chatbot.retriever.load_corpus``. They are never
positions in a filtered list: withholding entries for the cold-start
simulation shifts list positions but leaves ``FaqEntry.index`` stable.
"""
