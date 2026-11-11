# Project Report: Heart Disease Prediction using Logistic Regression

## Abstract

This project develops an educational binary classification application to estimate whether input attributes are associated with the presence of heart disease in the supplied dataset. It uses a Scikit-learn preprocessing and Logistic Regression pipeline, hyperparameter tuning, evaluation metrics, a FastAPI inference service, and a Streamlit interface. Docker is provided for reproducible deployment. MLflow is used for experiment tracking and model artifact management. The project demonstrates an end-to-end machine learning workflow, from data preprocessing and model development to deployment and monitoring.

## 1. Problem Statement

The objective of this project is to develop a reproducible machine-learning application that predicts the binary heart-disease target from 13 patient attributes and exposes the trained model through an API and graphical interface.

The application uses Logistic Regression to classify input records into two categories: disease indicators absent (0) and disease indicators present (1). The output is intended exclusively for educational and experimental purposes and must not be considered a medical diagnosis.

## 2. Dataset Understanding

The supplied CSV contains 303 records and 14 columns: 13 input features and one target variable. The dataset is loaded without a header.

The original target variable contains five levels (0, 1, 2, 3, and 4). For binary classification, the target is converted using the following mapping:

- Original target 0 is mapped to 0 (No disease indicators).
- Original targets 1, 2, 3, and 4 are mapped to 1 (Disease indicators present).

The `?` markers in the `ca` and `thal` features are converted into missing values during data loading. These missing values are subsequently handled through the preprocessing pipeline.

## 3. Data Preprocessing

The dataset is divided into training and testing subsets using a stratified 80/20 train-test split, with a fixed random state of 42.

The preprocessing operations include:

- **Numerical features:** Missing values are handled using median imputation, followed by standardization using StandardScaler.
- **Categorical features:** Missing values are handled using most-frequent imputation, followed by one-hot encoding using OneHotEncoder with `handle_unknown="ignore"`.

All preprocessing operations are integrated into a Scikit-learn ColumnTransformer and Pipeline. The transformations are fitted only on the training data within each cross-validation fold, reducing the possibility of data leakage.

## 4. Model Development

Logistic Regression is selected as the classification algorithm because it is suitable for binary classification and provides class probability estimates.

The model uses balanced class weights to account for the distribution of the target classes.

Hyperparameter optimization is performed using GridSearchCV with five-fold StratifiedKFold cross-validation. The search evaluates different regularization strengths (C) and solvers.

The candidate values are:

- Regularization strength (C): 0.01, 0.1, 1, 10, and 100.
- Solvers: `liblinear` and `lbfgs`.

Recall is selected as the optimization objective. This prioritizes identifying positive cases but may also increase false-positive predictions.

The best-performing configuration is selected through cross-validation and subsequently evaluated using the held-out test dataset.

## 5. Model Evaluation

The training process completed successfully using a stratified 80/20 train-test split.

- Total records: 303
- Training records: 242
- Testing records: 61
- Random state: 42

The following evaluation metrics were obtained on the held-out test dataset:

| Metric | Result |
|---|---:|
| Accuracy | 0.8852 |
| Precision | 0.8387 |
| Recall | 0.9286 |
| F1-score | 0.8814 |
| ROC-AUC | 0.9665 |
| Best CV Recall | 0.8099 |

The selected hyperparameters are:

- `classifier__C`: 0.1
- `classifier__solver`: liblinear

The evaluation results are saved in `metrics.json` and `classification_report.txt`. The confusion matrix and ROC curve are generated and saved as `confusion_matrix.png` and `roc_curve.png`, respectively.

### 5.1 Interpretation of Classification Errors

The model's predictions can result in two types of classification errors:

**False Positive (FP):**

A false positive occurs when the model predicts that disease indicators are present (class 1), while the actual dataset label indicates no disease (class 0). This represents an incorrect positive prediction.

**False Negative (FN):**

A false negative occurs when the model predicts that disease indicators are absent (class 0), while the actual dataset label indicates disease (class 1). This represents a missed positive case.

In a healthcare-related classification task, false negatives are particularly important because they represent cases in which the model fails to identify a positive instance. False positives are also important because they may result in unnecessary concern or additional assessment.

Since recall is used as the GridSearchCV optimization objective, the model selection process prioritizes identifying positive cases. However, improving recall may increase false-positive predictions.

Both types of errors must be considered when evaluating the model. The reported results are based on a small held-out dataset and do not establish clinical effectiveness or diagnostic reliability.

## 6. Deployment

The trained model is deployed through a FastAPI-based inference service and a Streamlit graphical user interface.

### 6.1 FastAPI

FastAPI provides the following endpoints:

- `POST /predict`: Accepts patient attributes, validates the input, and returns the predicted class, label, and estimated probability.
- `GET /health`: Reports the API status and whether the trained model has been successfully loaded.
- Interactive API documentation is available through the Swagger UI at `/docs`.

Pydantic validation is used to check incoming input values. Invalid requests are rejected with appropriate HTTP error responses.

### 6.2 Streamlit Interface

Streamlit provides a form-based graphical interface through which users can enter the 13 input attributes.

The interface communicates with the FastAPI prediction endpoint and displays the prediction result, estimated probability, and an educational disclaimer.

### 6.3 Docker Deployment

Docker is used to containerize the application for reproducible deployment.

Docker Compose manages two separate services:

- **API service:** Runs the FastAPI application on port 8000.
- **Streamlit service:** Runs the graphical interface on port 8501.

The API service includes a Docker health check to verify its availability.

## 7. Experiment Tracking and Model Artifacts

The project uses MLflow for experiment tracking and artifact management.

The training script records the following information:

- Selected hyperparameters.
- Evaluation metrics, including accuracy, precision, recall, F1-score, and ROC-AUC.
- Dataset information, including the number of records and dataset filename.
- The trained preprocessing and Logistic Regression pipeline.
- Evaluation metrics artifact.

The trained model pipeline is saved locally using Joblib as `heart_disease_pipeline.joblib`. It is also logged in MLflow as a model artifact, together with `metrics.json`.

MLflow provides an experiment dashboard through which the recorded runs, parameters, metrics, and artifacts can be inspected.

The experiment is named `heart-disease-logistic-regression`. This supports experiment reproducibility and provides a record of the model training process.

## 8. Monitoring and Maintenance

Monitoring and maintenance are important for maintaining application reliability and understanding model performance over time.

### 8.1 Operational Monitoring

The application incorporates basic logging and service-health monitoring.

The following monitoring mechanisms are implemented:

- **Prediction logging:** The FastAPI application records the predicted class and corresponding probability for each successful prediction.
- **Error logging:** Application errors and prediction-processing exceptions are recorded in the application log.
- **Privacy-aware logging:** Individual patient feature values are not stored in prediction logs.
- **API health monitoring:** The `/health` endpoint reports the service status and whether the trained model has been successfully loaded.
- **Container monitoring:** Docker Compose provides a health check for the API container and allows the status of the API and Streamlit services to be inspected.
- **Experiment tracking:** MLflow records model parameters, evaluation metrics, and the saved model artifact.

### 8.2 Model Performance Monitoring

The current implementation evaluates model performance using a held-out test dataset. The following metrics are recorded:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

The evaluation results are saved in `reports/metrics.json` and logged through MLflow.

For future maintenance, the model should be periodically evaluated using newly collected, appropriately labeled, and representative data. Changes in model performance should be examined before deciding whether retraining is necessary.

### 8.3 Data Drift Monitoring

Data drift occurs when the distribution of incoming data changes compared with the distribution of the training dataset. Such changes may affect model performance.

As a future enhancement, the distributions of incoming patient attributes can be compared with those of the original training dataset to identify potential data drift.

A detected distribution change should be investigated and followed by model evaluation using representative labeled data. Distribution changes alone should not automatically trigger model retraining.

### 8.4 Model Maintenance and Retraining Strategy

Model maintenance is necessary to ensure that the model remains suitable for the data on which it is evaluated.

The following maintenance strategy is proposed:

1. Periodically monitor API availability, application logs, and prediction errors.
2. Collect appropriately consented and anonymized data for future evaluation, where permitted.
3. Examine changes in incoming feature distributions to identify potential data drift.
4. Evaluate model performance using newly verified and representative ground-truth labels.
5. Consider retraining when significant data drift or measurable performance degradation is identified.
6. Train and evaluate a replacement model before deployment.
7. Compare the replacement model with the existing model using appropriate evaluation metrics.
8. Maintain model artifacts, parameters, and experiment records using MLflow and Git for reproducibility and version management.

The current implementation provides basic operational logging, health checks, and experiment tracking. Automated data drift detection, continuous production performance monitoring, and automatic model retraining are not implemented and are considered future enhancements.

## 9. Limitations and Ethics

The project has several limitations that must be considered when interpreting its results.

### 9.1 Dataset Limitations

The dataset contains only 303 records and may be relatively old. It may not adequately represent the characteristics of the population in which the application could potentially be used.

The model is evaluated using one fixed train-test split. Therefore, the reported metrics may vary with different data samples and do not establish generalization to external populations.

### 9.2 Medical and Predictive Limitations

The model is developed for educational and experimental purposes only. Its predicted probabilities are model-estimated probabilities and must not be interpreted as clinical risk estimates.

The application must not be used for medical diagnosis, treatment decisions, or independent clinical assessment. External validation and clinical review would be required before considering any real-world healthcare application.

### 9.3 Privacy and Data Protection

Patient-related information can be sensitive. The application intentionally avoids storing individual patient feature values in prediction logs. Only the predicted class and corresponding probability are recorded for successful predictions.

Any future collection, storage, or processing of health-related information must follow appropriate privacy, security, consent, and access-control practices. Sensitive patient information should not be exposed through application logs or publicly accessible repositories.

### 9.4 Fairness and Bias

The model may produce different prediction performance across demographic or population subgroups because of limitations in the training dataset or differences in feature distributions.

Before any real-world application, model performance should be assessed across relevant demographic groups to identify potential biases and disparities. Fairness evaluation, representative external validation, and appropriate clinical expert review would be necessary.

## 10. Conclusion

The project demonstrates a reproducible end-to-end machine-learning workflow for educational heart disease classification using Logistic Regression.

The workflow includes dataset loading, preprocessing, hyperparameter optimization, model evaluation, artifact saving, MLflow experiment tracking, FastAPI-based inference, Streamlit-based interaction, and Docker deployment.

Git and GitHub are used for source-code version control, while MLflow maintains experiment records and model artifacts.

The application demonstrates the integration of machine learning with API development, graphical interfaces, experiment tracking, and containerized deployment. However, it remains an educational prototype and requires further validation, fairness assessment, privacy safeguards, and clinical review before any practical healthcare use.