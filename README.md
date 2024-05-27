# fraud-detection-anomaly-classification

# Credit Card Fraud Detection

## Overview
This project aims to detect fraudulent transactions in a credit card dataset using various machine learning techniques, including anomaly detection and classification algorithms. The primary goal is to build a model that accurately identifies fraudulent transactions.

## Dataset
The dataset used in this project is sourced from Kaggle. You can download it from the following link:
[Credit Card Fraud Detection Dataset] (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Dataset Setup
Download the dataset from Kaggle and place it in the data/ directory.
Unzip the downloaded file if necessary.
Project Steps
1. Data Exploration
Import necessary libraries.
Load the dataset into a Pandas DataFrame.
Display the first few rows and get information about the dataset.
Get summary statistics and check the distribution of the target variable (Class).
2. Data Preprocessing
Check for missing values.
Handle class imbalance using Synthetic Minority Over-sampling Technique (SMOTE).
3. Anomaly Detection
Implement and evaluate various anomaly detection algorithms:
Isolation Forest
Local Outlier Factor
One-Class SVM
Elliptic Envelope
Train the models and make predictions.
Evaluate the models using precision, recall, F1-score, and ROC AUC.
4. Feature Engineering
Apply normalization and scaling to features.
Use Principal Component Analysis (PCA) for dimensionality reduction.
Concatenate original features with PCA-transformed features.
5. Model Tuning
Perform hyperparameter tuning using Grid Search with cross-validation on the Isolation Forest model.
6. Classification Models
Train and evaluate basic classification algorithms:
Logistic Regression
Support Vector Machine (SVM)
Random Forest
7. Model Evaluation
Evaluate the classification models using accuracy, precision, recall, F1-score, and ROC AUC.
Compare the performance of different models and select the best-performing one.
Results
The Random Forest model achieved the highest performance with the following metrics:

Precision: 97.40%
Recall: 76.53%
F1-score: 85.71%
ROC AUC: 88.26%
Conclusion
The project successfully implemented various machine learning techniques to detect fraudulent transactions. The Random Forest model showed the best performance in terms of precision, recall, F1-score, and ROC AUC.

Contributing
We welcome contributions, suggestions, and feedback. Please submit issues or pull requests.

License
This project is licensed under the MIT License.

Acknowledgments
The dataset is sourced from Kaggle: Credit Card Fraud Detection Dataset
