# Part 3 – Advanced Modeling: Ensembles, Hyperparameter Tuning, and Full ML Pipeline

## Overview

This project extends the machine learning models developed in Part 2 by introducing advanced ensemble techniques, systematic hyperparameter tuning, cross-validation, learning curve analysis, and model serialization.

The objective is to identify the most robust classification model for predicting income using the Adult Census Income dataset and package the final model into a reusable Scikit-learn Pipeline.

---

# Dataset

* Dataset: Adult Census Income Dataset
* File Used: `cleaned_adult.csv`
* Target Variable: `income`

---

# Libraries Used

* pandas
* numpy
* matplotlib
* joblib
* scikit-learn

---

# Task 1 – Decision Tree Baseline

A DecisionTreeClassifier with default parameters (`max_depth=None`) was trained using the scaled training data.

### Results

* Training Accuracy: **1.0000**
* Test Accuracy: **0.8119**

### Analysis

The model achieved perfect training accuracy but significantly lower testing accuracy, indicating **overfitting**. An unconstrained decision tree continues splitting until nearly every training sample is classified correctly, causing it to memorize the training data instead of learning general patterns.

Decision Trees are considered **high-variance models** because small changes in the training data can produce very different tree structures and predictions.

---

# Task 2 – Controlled Decision Tree

A second Decision Tree was trained using:

* max_depth = 5
* min_samples_split = 20

### Results

* Training Accuracy: **0.8533**
* Test Accuracy: **0.8486**

### Analysis

Limiting the maximum depth prevents the tree from becoming excessively complex, while increasing the minimum number of samples required to split reduces unnecessary branches.

Compared with the unconstrained tree, the controlled model reduced overfitting and achieved better generalization on unseen data.

---

# Task 3 – Gini vs Entropy

Two Decision Tree models were trained using different impurity measures.

### Gini Criterion

Test Accuracy: **0.8483**

Formula:

Gini Impurity =

**1 − Σ(pi²)**

### Entropy Criterion

Test Accuracy: **0.8440**

Formula:

**Entropy = −Σ(pi log₂ pi)**

### Interpretation

A Gini impurity value of **0** means every sample in the node belongs to exactly one class, representing a perfectly pure node.

Both impurity measures produced similar predictive performance, although Gini was slightly better on this dataset.



# Task 4 – Random Forest

A Random Forest Classifier was trained using the following parameters:

* n_estimators = 100
* max_depth = 10
* random_state = 42

The model was evaluated using Accuracy and ROC-AUC.

The five most important features were extracted using the `feature_importances_` attribute.

The top contributing features included variables such as **capital.gain**, **marital.status_Married-civ-spouse**, **education.num**, **marital.status_Never-married**, and **age**.

## Feature Importance

Random Forest computes feature importance by measuring the **average reduction in Gini impurity** contributed by each feature across all trees in the forest.

Unlike Linear Regression coefficients, feature importance values do not indicate whether a feature has a positive or negative relationship with the target. Instead, they measure how useful each feature is for splitting the data and improving prediction accuracy.

## Bagging

Random Forest uses **Bootstrap Aggregating (Bagging)**. Multiple decision trees are trained on different bootstrap samples of the training dataset, and their predictions are combined through majority voting. Bagging reduces variance, improves model stability, and helps prevent overfitting compared with a single Decision Tree.

---

# Task 4a – Gradient Boosting

A Gradient Boosting Classifier was trained using:

* n_estimators = 100
* learning_rate = 0.1
* random_state = 42

Training Accuracy and ROC-AUC were calculated to evaluate performance.

Gradient Boosting builds trees sequentially, where each new tree attempts to correct the errors made by previous trees. Unlike Random Forest, which trains trees independently, Gradient Boosting improves performance by learning from previous mistakes.

---

# Task 5 – Feature Ablation Study

The five least important features identified by the Random Forest were removed from the dataset.

A second Random Forest model with the same hyperparameters was trained using the reduced feature set.

The ROC-AUC scores of both models were compared.

If the reduced model achieves similar performance, the removed features contribute little predictive value and can be considered noise. Removing such features simplifies the model, reduces computational cost, and may improve interpretability while maintaining similar predictive performance.

---

# Task 6 – Cross-Validated Model Comparison

A **5-fold Stratified Cross Validation** was performed using ROC-AUC as the evaluation metric.

The following models were evaluated:

| Model                    | Mean ROC-AUC | Standard Deviation |
| ------------------------ | -----------: | -----------------: |
| Logistic Regression      |       0.9060 |             0.0032 |
| Controlled Decision Tree |       0.8859 |             0.0042 |
| Random Forest            |       0.9121 |             0.0033 |
| Gradient Boosting        |       0.9216 |             0.0039 |

Cross-validation provides a more reliable estimate of model performance because every observation is used for both training and validation across different folds. This reduces dependence on a single train-test split and produces a more robust estimate of generalization performance.



# Task 7 – Hyperparameter Tuning using GridSearchCV

A complete Scikit-learn Pipeline was created consisting of:

* SimpleImputer (strategy = "median")
* StandardScaler
* RandomForestClassifier

GridSearchCV was used to evaluate multiple combinations of Random Forest hyperparameters. The best parameter combination was selected based on the highest cross-validated ROC-AUC score.

Grid Search evaluates every possible parameter combination, making it thorough but computationally expensive. Randomized Search evaluates only a subset of combinations, making it faster but less exhaustive.

---

# Task 8 – Manual Learning Curve

The best pipeline obtained from GridSearchCV was trained using progressively larger portions of the training dataset:

* 20%
* 40%
* 60%
* 80%
* 100%

For each training fraction, both the training ROC-AUC and test ROC-AUC were computed.

The learning curve helps determine whether the model is limited by insufficient training data or by model capacity.

If both curves converge as training data increases, the model is generalizing well. If a large gap remains between training and testing performance, additional regularization or model simplification may be beneficial.

---

# Task 9 – Model Serialization

The best machine learning pipeline was saved using:

```python
joblib.dump(best_pipeline, "best_model.pkl")
```

The saved model can later be loaded using:

```python
import joblib

loaded_model = joblib.load("best_model.pkl")

sample_data = X.head(2)

predictions = loaded_model.predict(sample_data)

print(predictions)
```

Saving the trained model allows future predictions without retraining, making deployment faster and more efficient.

---

# Summary Comparison

| Model                    | 5-Fold CV Mean AUC | Performance |
| ------------------------ | -----------------: | ----------- |
| Logistic Regression      |             0.9060 | Good        |
| Controlled Decision Tree |             0.8859 | Good        |
| Random Forest            |             0.9121 | Very Good   |
| **Gradient Boosting**    |         **0.9216** | **Best**    |

---

# Recommended Model

Based on the evaluation results, **Gradient Boosting** is recommended as the final model for deployment. It achieved the highest cross-validated ROC-AUC among all evaluated models, demonstrating strong predictive performance and good generalization. Compared with a single Decision Tree, it is less prone to overfitting, and compared with Logistic Regression, it captures more complex relationships within the data. Overall, Gradient Boosting provides the best balance between accuracy, robustness, and generalization for the Adult Census Income dataset.

---

# Files Included

The repository contains:

* Question_3_Advanced_Modeling.ipynb
* cleaned_adult.csv
* best_model.pkl
* feature_importance.csv
* cv_results.csv
* learning_curve.csv
* README.md

---

# Conclusion

This project successfully implemented advanced machine learning techniques, including Decision Trees, Random Forest, Gradient Boosting, feature ablation, cross-validation, hyperparameter tuning, learning curve analysis, and model serialization. The experiments demonstrate the importance of ensemble learning, systematic model evaluation, and hyperparameter optimization in developing robust predictive models. The final Gradient Boosting model achieved the strongest overall performance and is recommended for deployment.

