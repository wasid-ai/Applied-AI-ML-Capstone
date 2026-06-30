# Supervised Machine Learning – Regression and Classification

## Project Overview

This project implements a complete supervised machine learning workflow using the Adult Census Income dataset. The objective is to build, train, and evaluate both a regression model and a binary classification model. The project includes data preprocessing, feature encoding, feature scaling, model training, evaluation, regularization, threshold analysis, and bootstrap validation while following best practices to prevent data leakage.

---

# Dataset

The cleaned dataset (`cleaned_adult.csv`) generated in Part 1 is used throughout this project.

The dataset contains demographic and employment-related information, including:

* Age
* Workclass
* Education
* Marital Status
* Occupation
* Relationship
* Race
* Sex
* Capital Gain
* Capital Loss
* Hours per Week
* Native Country
* Income

---

# Feature Matrix and Target Variables

## Feature Matrix (X)

The feature matrix consists of all predictor variables except the selected target variable.

## Regression Target (y_reg)

The regression target selected for this project is:

**hours.per.week**

This is a continuous numerical variable, making it suitable for Linear Regression and Ridge Regression.

## Classification Target (y_clf)

The binary classification target is:

**income**

Income is classified into two categories:

* Income ≤ 50K
* Income > 50K

These classes are used to train the Logistic Regression classifier.

---

# Data Preprocessing

## Handling Categorical Variables

The Adult dataset contains multiple categorical variables.

### Label Encoding

Label Encoding is suitable only when categories possess a meaningful order.

Example:

* Low → 0
* Medium → 1
* High → 2

This preserves the natural ranking among categories.

### One-Hot Encoding

Most categorical variables in this dataset have **no natural ordering**, such as:

* Workclass
* Occupation
* Marital Status
* Native Country

Therefore, One-Hot Encoding was applied.

The first dummy variable was dropped to avoid multicollinearity.

One-Hot Encoding prevents the model from incorrectly assuming that one category is numerically larger or smaller than another, which would happen if Label Encoding were used on unordered categories.

---

# Train-Test Split

The dataset was divided into:

* Training Set: 80%
* Testing Set: 20%

using:

* `random_state = 42`

This ensures reproducible experimental results.

---

# Feature Scaling

Feature scaling was performed using **StandardScaler**.

The scaler was fitted **only on the training data** and then applied to both the training and testing datasets.

This prevents **data leakage**.

If the scaler were fitted on the entire dataset before splitting, information from the testing data would influence the training process, resulting in overly optimistic evaluation metrics.



# Regression Models

## Linear Regression

A Linear Regression model was trained using the scaled training dataset to predict **hours.per.week**.

The model was evaluated using the following metrics:

* Mean Squared Error (MSE)
* R² Score

The coefficients of the trained model were extracted and matched with their corresponding feature names. The three features with the largest absolute coefficient values were identified to determine which variables had the greatest impact on the prediction.

### Coefficient Interpretation

A **positive coefficient** indicates that increasing the value of a feature increases the predicted target value while keeping all other features constant.

A **negative coefficient** indicates that increasing the feature value decreases the predicted target value.

Features with larger absolute coefficient values have a stronger influence on the prediction.

---

## Ridge Regression

A Ridge Regression model with **alpha = 1.0** was trained using the same training and testing datasets.

The performance of Ridge Regression was compared with Linear Regression using the R² score.

| Model             | Evaluation Metric |
| ----------------- | ----------------- |
| Linear Regression | R² Score          |
| Ridge Regression  | R² Score          |

Ridge Regression applies **L2 Regularization**, which shrinks large coefficient values and helps reduce overfitting.

The **alpha** parameter controls the amount of regularization.

* Small alpha → weaker regularization
* Large alpha → stronger regularization

Because Ridge penalizes excessively large coefficients, it often produces a more stable model than ordinary Linear Regression.

---

# Logistic Regression

The binary classification model was built using **Logistic Regression**.

The target variable (**income**) contains two classes:

* Income ≤ 50K
* Income > 50K

Before training, the class distribution was examined.

Since the dataset contains class imbalance, **class_weight = "balanced"** was used during model training. This automatically assigns higher importance to the minority class without modifying the original dataset.

The model was trained using:

* LogisticRegression
* max_iter = 1000
* class_weight = "balanced"

---

# Classification Evaluation

The classifier was evaluated using:

* Confusion Matrix
* Accuracy
* Precision
* Recall
* F1-Score
* ROC Curve
* Area Under Curve (AUC)

## Precision

Precision = TP / (TP + FP)

Precision measures how many predicted positive samples are actually positive.

## Recall

Recall = TP / (TP + FN)

Recall measures how many actual positive samples are correctly identified by the classifier.

For this income classification task, **Recall is considered more important**, because missing individuals who truly belong to the positive class (False Negatives) may be more costly than predicting a few additional positives.

---

# ROC Curve and AUC

The Receiver Operating Characteristic (ROC) Curve illustrates the trade-off between the True Positive Rate and False Positive Rate at different decision thresholds.

The Area Under the Curve (AUC) summarizes the classifier's ability to distinguish between the two classes.

* AUC = 1.0 indicates perfect classification.
* AUC = 0.5 indicates random guessing.

A higher AUC value indicates better classification performance and stronger class separation.



# Decision Threshold Sensitivity

The Logistic Regression model generated prediction probabilities using `predict_proba()`. Different decision thresholds (0.30, 0.40, 0.50, 0.60, and 0.70) were evaluated to understand their effect on Precision, Recall, and F1-score.

Lower thresholds generally increase Recall because more positive cases are detected, but they may reduce Precision by increasing False Positives.

Higher thresholds generally increase Precision because fewer samples are classified as positive, but they may reduce Recall by increasing False Negatives.

The threshold with the highest F1-score provides the best balance between Precision and Recall for this dataset.

---

# Logistic Regression Regularization

A second Logistic Regression model was trained with **C = 0.01**, representing stronger L2 regularization.

The performance of the baseline model (**C = 1.0**) and the strongly regularized model (**C = 0.01**) was compared using:

* Precision
* Recall
* AUC Score

The parameter **C** controls the inverse strength of regularization.

* Smaller C → Stronger Regularization
* Larger C → Weaker Regularization

Strong regularization reduces model complexity and may improve generalization by reducing overfitting. However, excessive regularization can reduce predictive performance if the model becomes too simple.

---

# Bootstrap Confidence Interval

A bootstrap experiment with **500 resamples** was performed to estimate the reliability of the AUC difference between the two Logistic Regression models.

For each bootstrap sample:

* Test observations were sampled with replacement.
* The AUC of both models was computed.
* The difference in AUC (C = 1.0 − C = 0.01) was recorded.

After 500 iterations, the following statistics were calculated:

* Mean AUC Difference
* Lower 95% Confidence Interval
* Upper 95% Confidence Interval

If the 95% confidence interval excludes zero, the performance difference between the two models is considered reliable. If the interval includes zero, the observed difference may not be statistically significant.

---

# Project Files

The repository contains the following files:

* Question_2_Supervised_Machine_Learning.ipynb
* cleaned_adult.csv
* README.md
* roc_curve.png
* ridge_vs_linear.csv
* logistic_comparison.csv
* bootstrap_auc_results.csv
* threshold_analysis.csv
* classification_report.csv
* confusion_matrix.csv
* top3_coefficients.csv
* model_results.json

---

# Conclusion

This project demonstrates a complete supervised machine learning workflow using the Adult Census Income dataset. Appropriate preprocessing techniques, feature encoding, train-test splitting, feature scaling, regression modelling, classification modelling, regularization, ROC analysis, threshold sensitivity analysis, and bootstrap validation were successfully implemented. The evaluation results show the effectiveness of the developed models and highlight the importance of proper preprocessing and rigorous model evaluation in building reliable machine learning solutions.
