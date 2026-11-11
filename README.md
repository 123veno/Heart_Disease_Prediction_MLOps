# Heart Disease Prediction using Logistic Regression

An end-to-end educational ML application using the supplied Cleveland-style heart disease CSV, Scikit-learn, FastAPI, Streamlit, optional MLflow tracking, and Docker.

> **Safety notice:** This is an educational prototype, not a medical device. Predictions and probabilities must not be used for diagnosis, treatment, or clinical decisions.

## 1. Dataset and target

The supplied CSV has 303 rows and 14 columns (13 features plus target), without a header. The `?` values in `ca` and `thal` are treated as missing values. The original target is mapped as `0 -> 0` (no disease) and `1, 2, 3, 4 -> 1` (disease present). This creates a binary classification task.

Dataset provenance: local `data/heartdisease.csv` supplied for this project. Verify the dataset's source, license, and permitted use before public redistribution. The file format and feature definitions are consistent with the commonly used UCI Cleveland Heart Disease dataset, but the provenance of the supplied copy should be documented by the student/instructor.

## 2. Project structure

- `data/heartdisease.csv`: input dataset
- `src/train.py`: data loading, preprocessing pipeline, tuning, evaluation, artifact saving
- `models/`: trained pipeline and metadata (created by training)
- `reports/`: metrics, classification report, plots
- `api/main.py`: FastAPI prediction API and health endpoint
- `app/streamlit_app.py`: Streamlit interface
- `logs/`: training and prediction logs
- `Dockerfile`, `docker-compose.yml`: container setup

## 3. Local setup

Use Python 3.10 or 3.11.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python -m src.train
```

Training uses a stratified 80/20 train-test split with random state 42. Imputation, one-hot encoding, and scaling are fitted inside the training pipeline, preventing test-set information from leaking into preprocessing. GridSearchCV uses 5-fold stratified cross-validation and recall scoring. The final test set is held out until final evaluation.

The training command creates:
- `models/heart_disease_pipeline.joblib`
- `models/model_metadata.json`
- `reports/metrics.json`
- `reports/classification_report.txt`
- `reports/confusion_matrix.png`
- `reports/roc_curve.png`
- `logs/training.log`

## 4. Evaluation

Metrics include accuracy, precision, recall, F1-score, ROC-AUC, classification report, confusion matrix, and ROC curve. False negatives may be particularly concerning in a screening context because a person with disease indicators could be classified as negative. False positives can cause unnecessary worry or follow-up. This trade-off must be interpreted cautiously; this dataset is small and not representative of all populations.

## 4A. Results from the included training run

The training script completed on the supplied dataset using a stratified 80/20 split (242 train, 61 test). Metrics: accuracy **0.885**, precision **0.839**, recall **0.929**, F1 **0.881**, ROC-AUC **0.966**. Best parameters: `{'classifier__C': 0.1, 'classifier__solver': 'liblinear'}`. Full results and plots are in `reports/`.

These are preliminary results on a small dataset and are not evidence of clinical validity.

## 5. Run the API

Train first, then:

```bash
uvicorn api.main:app --reload
```

- Health check: http://127.0.0.1:8000/health
- Interactive API docs: http://127.0.0.1:8000/docs
- Prediction endpoint: `POST /predict`

Example JSON:

```json
{
  "age": 55, "sex": 1, "cp": 2, "trestbps": 140,
  "chol": 220, "fbs": 0, "restecg": 0, "thalach": 150,
  "exang": 0, "oldpeak": 1.5, "slope": 2, "ca": 0, "thal": 3
}
```

## 6. Run Streamlit

In another terminal:

```bash
streamlit run app/streamlit_app.py
```

For local execution, change the sidebar API URL to `http://127.0.0.1:8000`. Open http://localhost:8501.

## 7. Docker

Train the model locally first so the model artifact exists in `models/`, then build and run:

```bash
docker compose up --build
```

Open:
- Streamlit: http://localhost:8501
- API docs: http://localhost:8000/docs
- API health: http://localhost:8000/health

Stop with `docker compose down`.

## 8. Experiment tracking

If MLflow is installed (included in requirements), training logs parameters, metrics, dataset filename/row count, model, and metrics artifact to the `heart-disease-logistic-regression` experiment. View runs with:

```bash
mlflow ui
```

Then open http://127.0.0.1:5000. If MLflow logging is unavailable, training still completes and local metrics/artifacts are saved.

## 9. Logging, monitoring, and maintenance

- Training events and metrics are written to `logs/training.log`.
- Prediction class and estimated probability are logged to `logs/prediction.log`; patient feature values are deliberately not logged.
- `/health` reports service and model-load status.
- Monitor API errors, request volume, prediction-class proportions, and latency. Where validated labels later become available, periodically evaluate recall, precision, F1, and ROC-AUC on a representative labeled sample.
- Check input distributions and missing/unknown category rates for data drift. Retrain only when a meaningful drift or sustained performance degradation is confirmed, using a newly versioned and quality-checked dataset. Keep a holdout test set and compare candidate models before replacing the deployed artifact.

## 10. Limitations and responsible use

- Only 303 rows are available; estimates may be unstable and generalization is limited.
- The dataset may be old and may not represent current clinical practice or the intended population.
- `class_weight="balanced"` and recall-focused tuning reflect a modeling choice, not clinical validation.
- The displayed class-1 probability is not a calibrated clinical risk estimate.
- The app does not provide feature-level clinical explanations.
- Do not enter identifiable patient information. This prototype is not suitable for real patient care.

## 11. Git version control

```bash
git init
git add .
git commit -m "Initial heart disease prediction application"
```

Do not commit real patient data or private information to a public repository. Check dataset licensing before publishing the CSV.

## 12. Submission checklist

- [ ] Run `python -m src.train` successfully.
- [ ] Confirm model and report artifacts were created.
- [ ] Start FastAPI and test `/health` and `/docs`.
- [ ] Start Streamlit and make a test request.
- [ ] Run `docker compose up --build` if Docker is available.
- [ ] Capture screenshots of the UI, API docs, metrics, and Docker services.
- [ ] Include the generated metrics in the report and explain limitations.
