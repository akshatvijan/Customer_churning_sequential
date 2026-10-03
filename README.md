# Customer Churn & Retention

FastAPI serves churn predictions, K-Means segments and retention recommendations from a saved model. Gradio calls the backend over HTTP. The Python request/response models define the OpenAPI contract.

## Run locally

Python 3.11+ is required; this setup was run with Python 3.13.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Place the team's `customer_churn_1M.csv` in `dataset/`. The CSV and generated model artifacts stay local and are excluded from Git.

```bash
# Offline export: stratified 20,000-row sample, 80/20 training/holdout split.
python -m scripts.export_model

# Starts the API and mounted Gradio interface. Training does not run on startup.
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

- Gradio: <http://127.0.0.1:8000/ui/>
- Swagger UI: <http://127.0.0.1:8000/docs>
- ReDoc: <http://127.0.0.1:8000/redoc>
- Live OpenAPI contract: <http://127.0.0.1:8000/openapi.json>
- Readiness: <http://127.0.0.1:8000/health>

The default serving model is the existing Random Forest configuration. This is a configurable example, not a claim that Random Forest won a model comparison. Model information includes actual held-out metrics, sample sizes and training time stamp.

Run all commands below from the repository root while the virtual environment is active. Keep `uvicorn` running in one terminal and use a second terminal for the `curl` commands. If you open `/health` before exporting the artifact, it returns HTTP 503; export the artifact and restart the server.

## OpenAPI contract and endpoints

The versioned contract is [`docs/openapi.json`](docs/openapi.json), generated directly from the typed Pydantic schemas in `app/schemas.py` and FastAPI routes. Refresh it after changing the API:

```bash
python -m scripts.export_openapi
```

| Method | Endpoint | Result |
| --- | --- | --- |
| GET | `/health` | Model readiness; HTTP 503 if unavailable |
| GET | `/model-info` | Model identity, held-out metrics, global importance and segment profiles |
| POST | `/predict-churn` | Probability, predicted class and risk level |
| POST | `/segment-customer` | K-Means segment ID and training-population profile |
| POST | `/recommend-retention` | Recommendation for supplied customer and probability |
| POST | `/customer-strategy` | Prediction, segment, value category, action, reward, cost and rule reasoning |
| POST | `/customer-strategy/batch` | 1–1,000 strategies, distributions and scenario ROI |

Prediction and segmentation accept one customer object. The batch body is `{"customers": [...]}`. Retention accepts `{"customer": {...}, "churn_probability": 0.82}`. Exact required fields, response schemas, operation IDs and errors are in Swagger and the exported contract.

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/customer-strategy \
  -H 'Content-Type: application/json' --data-binary @examples/customer.json
curl -s http://127.0.0.1:8000/customer-strategy/batch \
  -H 'Content-Type: application/json' --data-binary @examples/batch.json
curl -s http://127.0.0.1:8000/model-info
```

Required customer fields: `customer_id`, `tenure`, `monthlycharges`, `totalcharges`. Supply the other known dataset attributes for better-informed predictions; omitted optional attributes use the fitted training imputations. An example is in [`examples/customer.json`](examples/customer.json).

The API also accepts `monthly_charges`, `total_charges`, `complaints`, and `service_calls` as aliases of `monthlycharges`, `totalcharges`, `num_complaints`, and `num_service_calls`. Supplying both spellings is rejected. Unknown fields, `churn`, non-finite numeric values and invalid ranges return HTTP 422 with structured details. The Gradio CSV importer explicitly removes the historical `churn` label before calling the API. Dates use ISO 8601.

A missing or incompatible artifact returns HTTP 503; export the model and restart the server. There are no fallback or fabricated predictions.

## API demo: inputs for every condition

The JSON examples and commands below work with the default model export. For `POST` requests use `Content-Type: application/json`. Swagger at <http://127.0.0.1:8000/docs> also lets you paste the same JSON into each endpoint's **Try it out** form. Open `/model-info` first to see the model actually loaded and its saved customer-value thresholds.

### 1. Check the server and inspect model results

```bash
curl -i http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/model-info
curl -s http://127.0.0.1:8000/openapi.json
```

Expected: `/health` reports `"status":"ready"`; `/model-info` reports the selected classifier, held-out metrics, feature importance, segment profiles and `value_thresholds`. `/openapi.json` is the machine-readable API contract.

### 2. Call every customer endpoint

The sample customer in [`examples/customer.json`](examples/customer.json) has `customer_id=C10291`, `tenure=14`, `monthlycharges=89.50`, `totalcharges=1253.00`, `num_complaints=3`, `num_service_calls=5`, and `customer_satisfaction=4`. These fields are valid for prediction, segmentation and the combined strategy endpoint:

```bash
curl -s http://127.0.0.1:8000/predict-churn \
  -H 'Content-Type: application/json' --data-binary @examples/customer.json

curl -s http://127.0.0.1:8000/segment-customer \
  -H 'Content-Type: application/json' --data-binary @examples/customer.json

curl -s http://127.0.0.1:8000/customer-strategy \
  -H 'Content-Type: application/json' --data-binary @examples/customer.json

curl -s http://127.0.0.1:8000/customer-strategy/batch \
  -H 'Content-Type: application/json' --data-binary @examples/batch.json
```

`/predict-churn` returns a model-generated probability and risk. `/segment-customer` returns a fitted K-Means segment. `/customer-strategy` combines those results with a value tier and reward. The batch endpoint returns results in input order, risk/segment counts and an illustrative campaign ROI. You cannot force a particular predicted probability by entering a chosen customer: the fitted model calculates it. To check every **risk and reward rule deterministically**, use `/recommend-retention` with a supplied probability as shown next.

### 3. Cover all nine risk × customer-value reward rules

`/recommend-retention` accepts an explicit `churn_probability`; it does not call the classifier. Use this body shape, changing `churn_probability` and `totalcharges` according to the table:

```json
{
  "customer": {
    "customer_id": "VIVA-01",
    "tenure": 14,
    "monthlycharges": 89.50,
    "totalcharges": 1000.00
  },
  "churn_probability": 0.55
}
```

The **current local artifact** has value cutoffs near `680.03` and `2023.73`, so `100`, `1000` and `3000` represent Low, Medium and High value respectively. After retraining, read `/model-info` and choose one value below its `low` cutoff, one strictly between `low` and `high`, and one above `high`. Risk cutoffs stay fixed at `0.40` and `0.70`.

| Probability | Total charges | Expected risk | Expected value | Expected action | Expected reward cost |
| ---: | ---: | --- | --- | --- | ---: |
| `0.20` | `100` | Low | Low | Monitor customer | `0` |
| `0.20` | `1000` | Low | Medium | Engagement campaign | `50` |
| `0.20` | `3000` | Low | High | Loyalty retention | `100` |
| `0.55` | `100` | Medium | Low | Low-cost re-engagement | `50` |
| `0.55` | `1000` | Medium | Medium | Targeted retention campaign | `100` |
| `0.55` | `3000` | Medium | High | Priority retention | `200` |
| `0.85` | `100` | High | Low | Cost-effective win-back | `50` |
| `0.85` | `1000` | High | Medium | Urgent retention campaign | `200` |
| `0.85` | `3000` | High | High | High-priority retention | `300` |

For one case, paste this directly into a terminal:

```bash
curl -s http://127.0.0.1:8000/recommend-retention \
  -H 'Content-Type: application/json' \
  -d '{"customer":{"customer_id":"VIVA-HIGH-HIGH","tenure":14,"monthlycharges":89.5,"totalcharges":3000},"churn_probability":0.85}'
```

To request all nine cases, run this shell loop. It prints one JSON response per combination:

```bash
for probability in 0.20 0.55 0.85; do
  for charges in 100 1000 3000; do
    curl -s http://127.0.0.1:8000/recommend-retention \
      -H 'Content-Type: application/json' \
      -d "{\"customer\":{\"customer_id\":\"VIVA-$probability-$charges\",\"tenure\":14,\"monthlycharges\":89.5,\"totalcharges\":$charges},\"churn_probability\":$probability}"
    printf '\n'
  done
done
```

Boundary values are useful in a viva: `0.00` and `0.40` must be **Low** risk; `0.70` must be **Medium**; `1.00` must be **High**. Set `churn_probability` to each value in the sample body to demonstrate the exact rule boundaries.

### 4. Show aliases, missing optional fields and batch input

This minimal customer is valid: the fitted preprocessing pipeline fills omitted optional attributes. `monthly_charges` and `total_charges` demonstrate aliases. An unseen category such as `"contract":"new_plan"` is accepted by the fitted one-hot encoder, which ignores unknown categories.

```bash
curl -s http://127.0.0.1:8000/customer-strategy \
  -H 'Content-Type: application/json' \
  -d '{"customer_id":"MINIMAL-01","tenure":14,"monthly_charges":89.5,"total_charges":1253}'

curl -s http://127.0.0.1:8000/customer-strategy \
  -H 'Content-Type: application/json' \
  -d '{"customer_id":"NEW-PLAN","tenure":14,"monthlycharges":89.5,"totalcharges":1253,"contract":"new_plan"}'

curl -s http://127.0.0.1:8000/customer-strategy/batch \
  -H 'Content-Type: application/json' \
  -d '{"customers":[{"customer_id":"A","tenure":14,"monthlycharges":89.5,"totalcharges":1253},{"customer_id":"B","tenure":60,"monthlycharges":110,"totalcharges":5000}]}'
```

The API accepts **1–1,000** records per batch. The same customer receives the same model score, segment and value tier whether sent alone or inside a batch. Gradio's Batch tab also accepts a UTF-8 customer CSV up to **2 MB**, with these column names as headers, or a JSON array; it offers a scored CSV download.

### 5. Check documented error conditions

These inputs should return **HTTP 422** and a structured `detail` list. Add `-i` to see the HTTP status and response body:

```bash
# Missing the required totalcharges field
curl -i http://127.0.0.1:8000/customer-strategy \
  -H 'Content-Type: application/json' \
  -d '{"customer_id":"BAD-1","tenure":14,"monthlycharges":89.5}'

# Negative monthly charges
curl -i http://127.0.0.1:8000/customer-strategy \
  -H 'Content-Type: application/json' \
  -d '{"customer_id":"BAD-2","tenure":14,"monthlycharges":-1,"totalcharges":1253}'

# Both spellings of one field in the same request
curl -i http://127.0.0.1:8000/customer-strategy \
  -H 'Content-Type: application/json' \
  -d '{"customer_id":"BAD-3","tenure":14,"monthlycharges":89.5,"monthly_charges":89.5,"totalcharges":1253}'

# Training target is forbidden in prediction input
curl -i http://127.0.0.1:8000/predict-churn \
  -H 'Content-Type: application/json' \
  -d '{"customer_id":"BAD-4","tenure":14,"monthlycharges":89.5,"totalcharges":1253,"churn":1}'

# Probability outside 0–1
curl -i http://127.0.0.1:8000/recommend-retention \
  -H 'Content-Type: application/json' \
  -d '{"customer":{"customer_id":"BAD-5","tenure":14,"monthlycharges":89.5,"totalcharges":1253},"churn_probability":1.2}'

# Empty batch
curl -i http://127.0.0.1:8000/customer-strategy/batch \
  -H 'Content-Type: application/json' -d '{"customers":[]}'
```

To see the **HTTP 503** unavailable-model condition, stop Uvicorn and start an API-only instance with a nonexistent artifact path, then call `/health` or a prediction endpoint. Do not overwrite your working artifact:

```bash
CHURN_ENABLE_UI=0 CHURN_MODEL_PATH=/tmp/nonexistent-churn-model.joblib \
  uvicorn app.main:app --host 127.0.0.1 --port 8000

curl -i http://127.0.0.1:8000/health
curl -i http://127.0.0.1:8000/predict-churn \
  -H 'Content-Type: application/json' --data-binary @examples/customer.json
```

Stop that instance and restart Uvicorn normally for the successful demo.

## Gradio

The interface provides an individual customer form with optional JSON attributes, batch JSON/CSV scoring (up to 1,000 records and 2 MB), downloadable results, risk/segment charts, retention scenario ROI, and model information. Every model result comes from the FastAPI endpoints, using `httpx`.

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `CHURN_MODEL_PATH` | Repository `artifacts/churn_bundle.joblib` | Trusted, locally generated artifact to load once at startup |
| `CHURN_API_URL` | `http://127.0.0.1:8000` | Backend URL used by Gradio callbacks |
| `CHURN_ENABLE_UI` | `1` | Set `0` for an API-only process |

When changing the Uvicorn port, also set the client URL, for example:

```bash
CHURN_API_URL=http://127.0.0.1:8080 uvicorn app.main:app --host 127.0.0.1 --port 8080
```

This is a local demo with no authentication. The commands bind to loopback. Only load trusted joblib files: the serialization format can execute Python during loading. Artifacts require the same scikit-learn version used to train them; regenerate after a dependency upgrade.

## Supervised package

The supervised code follows the same organization as `unsupervised/`: individual model modules, shared helpers, a package initializer and one workflow entry point. Importing a module does not train a model or open plots.

```text
supervised/
  __init__.py
  data_preprocessing.py      # Reuses the team's feature engineering/preprocessing
  logistic_regression.py
  decision_tree.py
  random_forest.py
  svm_model.py
  xgboost_model.py
  model_evaluation.py        # Metrics, risk results, confusion matrices, importance
  supervised_learning.py     # run_supervised_analysis and CLI
  supervised_models.py       # Compatibility entry point
```

Existing classifier settings are preserved. Each model fits a fresh preprocessing pipeline to avoid fitted-state sharing. SVM now uses a bounded, stratified sample of at most 2,000 training rows instead of requiring 100,000; probability-enabled RBF SVM is expensive at the full dataset scale.

```bash
# Existing five-model workflow; XGBoost needs this optional dependency set.
python -m pip install -r requirements-supervised.txt
python -m supervised.supervised_learning --sample-size 20000

# Run selected models only; optionally add --show-plots.
python -m supervised.supervised_learning --models logistic_regression decision_tree random_forest

# Export any supported classifier for the API; then restart the backend.
python -m scripts.export_model --model decision_tree --sample-size 20000
# Other choices: logistic_regression, random_forest, svm, xgboost.
```

The workflow returns a `models` dictionary containing fitted pipelines, metrics, classification reports, confusion matrices, risk result tables and feature importance, plus a comparison DataFrame. The comparison is an experiment on a common holdout, not a separate validation-based model-selection procedure.

## Integration details

The artifact contains the complete fitted classifier pipeline, a median-imputation/scaling/K-Means pipeline, input schema, segment profiles and fixed customer-value thresholds. The same transformations and thresholds apply to individual and batch requests. K-Means segments are numbered groups; their profiles are displayed without inventing business names. PCA and DBSCAN remain available in the existing offline analysis modules.

Retention reuses the team's reward mappings. Low risk is probability <= 0.40; Medium is > 0.40 and <= 0.70; High is > 0.70. Customer-value cutoffs are the training population's 0.33 and 0.67 total-charge quantiles. They are never recomputed from one API customer or a submitted batch.

The retention scenario assumes 10% incremental retention, three additional months and 40% contribution margin, using the existing probability-weighted cost calculation. Customers with no recommended reward are excluded from campaign ROI. These assumptions are not validated causal uplift. Global importance and segment profiles are descriptive, not individual causal explanations. See the preserved [retention documentation](docs/retention-strategy.md).

## Verification and project scope

No pytest suite is added or run. API verification uses direct HTTP requests; results are recorded in [`docs/api-verification.md`](docs/api-verification.md). Model holdout metrics describe predictive quality, not automated software testing.

The PDF's broader deliverables (ANN comparison, complete EDA/data dictionary, final selection report and presentation) are outside this API/Gradio and supervised-structure change. Team analysis code remains available for that work.
