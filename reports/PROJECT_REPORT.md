# Project Report: Heart Disease Prediction using Logistic Regression

## Abstract
This project develops an educational binary classification application to estimate whether input attributes are associated with the presence of heart disease in the supplied dataset. It uses a Scikit-learn preprocessing and Logistic Regression pipeline, hyperparameter tuning, evaluation metrics, a FastAPI inference service, and a Streamlit interface. Docker is provided for reproducible deployment.

## 1. Problem statement
Develop a reproducible machine-learning application that predicts the binary heart-disease target from 13 patient attributes and exposes the model through an API and graphical interface. The output is educational only and is not a medical diagnosis.

## 2. Dataset understanding
The supplied CSV contains 303 records and 14 columns: 13 input features and one target. It is loaded without a header. The target is converted from five original levels to binary labels: original 0 becomes 0; original 1–4 become 1. The `?` markers in `ca` and `thal` are converted to missing values.

## 3. Preprocessing
A stratified 80/20 train-test split is created. Numerical features are median-imputed and standardized. Categorical features are most-frequent-imputed and one-hot encoded. All transformations are inside a Scikit-learn Pipeline and are fitted only on training folds, reducing data leakage.

## 4. Model development
Logistic Regression with balanced class weights is used. GridSearchCV evaluates candidate regularization strengths and solvers using five-fold StratifiedKFold cross-validation. Recall is the tuning objective; this prioritizes finding positive cases but may increase false positives.

## 5. Evaluation
The training run completed successfully with a stratified 80/20 split (242 training rows and 61 held-out test rows). Results for this fixed split:

| Metric | Result |
|---|---:|
| Accuracy | 0.8852 |
| Precision | 0.8387 |
| Recall | 0.9286 |
| F1-score | 0.8814 |
| ROC-AUC | 0.9665 |
| Best CV recall | 0.8099 |

Selected parameters: `{'classifier__C': 0.1, 'classifier__solver': 'liblinear'}`. See `metrics.json`, `classification_report.txt`, `confusion_matrix.png`, and `roc_curve.png`. Explain false negatives and false positives in the context of the educational screening task. These results are from one small dataset split and should not be interpreted as clinical performance.

## 6. Deployment
FastAPI provides `POST /predict`, `GET /health`, and interactive API documentation. Streamlit provides a form-based interface. Docker Compose runs the API and UI as separate services.

## 7. Experiment tracking and artifacts
The training script saves the fitted preprocessing/model pipeline, metadata, metrics, classification report, and evaluation plots. If MLflow is installed, it logs parameters, metrics, model, and metrics artifact.

## 8. Monitoring and maintenance
Operational errors and prediction summaries are logged without storing patient feature values. Monitor API health, errors, latency, prediction proportions, and data distributions. Evaluate on newly labeled representative data when available. Retrain only after confirming drift or performance degradation and validating the replacement model.

## 9. Limitations and ethics
The dataset is small (303 records), potentially old, and may not represent the population where the tool could be used. Model probabilities are not clinical risk estimates. The application must not be used for diagnosis or treatment decisions. Privacy, fairness, external validation, and clinical review would be required before any real-world use.

## 10. Conclusion
The project demonstrates a reproducible end-to-end ML workflow from data preprocessing and model evaluation to API/UI deployment. The application is a learning prototype and requires further validation before any practical healthcare use.
