# Project-Staircase-
AI Agent and Lead Scoring Application in the  Student Recruitment Sector 
This research proposes a prototype AI-powered student recruitment assistant integrating a
hybrid retrieval-based chatbot with a lead-scoring engine trained on real anonymised
counsellor assessments augmented by synthetic personas. The chatbot combines FAISS
similarity search with live web search fallback (via Tavily/SerpAPI) to provide comprehensive
responses to prospective student enquiries. 
The lead-scoring engine merges rule-based scoring derived from admissions literature with 
machine learning predictions trained on ahybrid dataset: real anonymised counsellor assessment forms (Cold/Good/Excellent labels)
from AIMS Education, augmented with synthetic personas grounded in observed feature
distributions. Data sources include publicly scraped university FAQs, anonymised counsellor
data with strict PII removal, and synthetic simulations. All personally identifiable information is
removed before analysis; only behavioural and qualification features are retained. Evaluation
measures chatbot retrieval accuracy using cosine similarity (target: >0.80), lead-scoring
F1-score on real counsellor data (target: >0.85), and hybrid score stability via sensitivity
analysis. The research demonstrates how meaningful AI systems can be developed using
secondary and carefully anonymised institutional data, anchoring predictive models in
real-world expert judgment while maintaining ethical compliance
