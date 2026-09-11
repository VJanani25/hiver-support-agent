# Hiver Support Agent

An interview-ready, retrieval-grounded customer-support agent for the Hiver SDE Intern take-home assignment.

The system accepts a customer message, predicts an intent, retrieves historically similar resolved conversations, drafts a reply using only retrieved evidence, and decides whether the message is safe to auto-handle or should be escalated.

## Why this repository is honest by default

The full Customer Support on Twitter dataset is not bundled. The repository includes a small, clearly labelled demo fixture so the code can be run immediately, but **demo metrics are not presented as benchmark results**. The real evaluation path requires:

1. downloading the Kaggle export;
2. inspecting the available brands/author accounts;
3. selecting one brand using the inspection output;
4. reviewing the generated golden-set candidates by hand; and
5. running the evaluation harness.

No API key is required for the default retrieval-grounded demo. An LLM judge can be added later, but the report does not claim LLM-judge results that have not been run and human-validated.

## Repository map

```text
hiver-support-agent/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .env.example
├── .gitignore
├── data/
│   ├── README.md
│   ├── sample/support_sample.csv
│   └── golden_set.csv
├── configs/
│   ├── intents.json
│   └── project.json
├── src/hiver_agent/
│   ├── agent.py
│   ├── data.py
│   ├── evaluation.py
│   ├── intent.py
│   ├── preprocessing.py
│   ├── retrieval.py
│   └── types.py
├── scripts/
│   ├── download_data.py
│   ├── prepare_data.py
│   ├── build_golden_set.py
│   ├── run_baselines.py
│   ├── run_agent.py
│   └── evaluate.py
├── tests/
│   └── test_core.py
└── reports/report.md
```

## Architecture

```text
customer message
      │
      ▼
normalization + tokenization
      │
      ├── intent classifier
      │     ├── majority baseline
      │     ├── keyword baseline
      │     └── TF-IDF + multiclass logistic regression
      │
      ├── retrieval index over resolved historical pairs
      │
      └── conservative policy
              ├── grounded reply from retrieved support response
              └── AUTO_HANDLE or ESCALATE with reason
```

The proposed model is intentionally understandable: a small TF-IDF vectorizer and multiclass softmax classifier implemented in pure Python, paired with sparse cosine retrieval. This makes the main behavior inspectable in an interview and avoids hiding the core logic behind a framework.

## Installation

```bash
cd hiver-support-agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The included smoke path uses only the Python standard library. `pytest` is used for the test command.

## Run the demo in under 15 minutes

```bash
cd hiver-support-agent
python scripts/prepare_data.py --demo
python scripts/build_golden_set.py --input data/processed/examples.csv --output data/golden_set.csv --size 200
python scripts/run_agent.py --demo --message "I was charged twice and need help"
python scripts/run_baselines.py --demo
python scripts/evaluate.py --demo
python -m pytest
```

The demo prints operational output and writes JSON artifacts under `reports/generated/`. It is a code-path check, not a claim about performance on the Kaggle benchmark.

### Example

```text
Input: I was charged twice and need help

Intent: billing_payment
Confidence: 0.91
Decision: AUTO_HANDLE
Reason: routine intent with strong classifier confidence and retrieved historical evidence
Reply: I’m sorry you were charged twice. Please share the transaction details so the support team can review the duplicate charge.
Evidence: 3 similar resolved conversations
```

The exact demo output may vary slightly as the deterministic sample is expanded.

## Use the real Customer Support on Twitter data

The assignment references [`thoughtvector/customer-support-on-twitter`](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter). Follow Kaggle's current download terms and keep the raw export outside Git.

```bash
python scripts/download_data.py --output data/raw
python scripts/prepare_data.py --input data/raw/tweets.csv --inspect-only
```

`--inspect-only` reports row counts, inbound/outbound counts, conversation-link coverage, likely support accounts, and common issue phrases. Use those findings to choose one brand/support account; do not pick a brand blindly.

```bash
python scripts/prepare_data.py \
  --input data/raw/tweets.csv \
  --brand-account <selected-support-account> \
  --output data/processed
```

The loader accepts the common Kaggle columns (`tweet_id`, `author_id`, `inbound`, `created_at`, `text`, `response_tweet_id`, and `in_response_to_tweet_id`) and fails with a useful message if the export has a different shape.

## Golden set workflow

Generate a deterministic candidate set:

```bash
python scripts/build_golden_set.py --input data/processed/examples.csv --output data/golden_set.csv --size 200
```

The generated rows contain `suggested_intent` and `suggested_action`, but the true `intent` and `expected_action` columns remain blank until a human reviews them. This is deliberate: predictions are not ground truth.

After review, fill:

```text
id,customer_message,context,intent,expected_action,notes
```

Then run:

```bash
python scripts/evaluate.py \
  --data data/processed/examples.csv \
  --golden data/golden_set.csv \
  --output reports/generated
```

## Evaluation

The harness reports:

- accuracy, macro F1, weighted F1;
- per-intent precision/recall/F1;
- confusion matrix;
- retrieval coverage and mean similarity;
- conservative escalation diagnostics;
- automated reply-grounding checks; and
- a structured human/LLM-judge comparison file when both ratings exist.

The two required baselines are:

1. majority intent classifier + short evidence reply;
2. keyword intent classifier + retrieval reply.

The proposed system is TF-IDF + multiclass logistic regression + retrieval + escalation policy. All result files are generated locally from the supplied data; the report never contains invented values.

## Environment variables

Copy `.env.example` if you add an external judge or model provider. The current demo does not require one.

```dotenv
OPENAI_API_KEY=
```

Never commit credentials.

## Engineering decisions

1. Use one brand/support account at a time so issue language and resolution style stay coherent.
2. Build customer/support pairs from response links rather than treating every tweet as independent.
3. Keep an explicit `other` intent for uncertainty.
4. Use sparse cosine retrieval to make evidence visible and auditable.
5. Reuse only retrieved support wording in the default reply generator.
6. Escalate when confidence, evidence, or risk signals are weak.
7. Keep the golden-set labels separate from model suggestions.
8. Use deterministic seeds and cached intermediate CSVs.

See [`reports/report.md`](reports/report.md) for the concise reviewer report, limitations, failure-analysis template, misleading-headline-number discussion, one-week plan, and decision log.

## Citation

The intended dataset is:

> Customer Support on Twitter, thoughtvector, Kaggle.  
> https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter

The code in this repository is original. Any future prompt or model provider added to the judge path should be documented here.

## Author

Janani Varadharajan — Computer Science student building practical Python and AI/ML systems.